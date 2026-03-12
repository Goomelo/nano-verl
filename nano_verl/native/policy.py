"""Policy model wrapper for the native engine."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CompletionSample:
    """One sampled completion plus the tensors needed for learning."""

    prompt_text: str
    completion_text: str
    full_token_ids: list[int]
    prompt_token_count: int
    completion_token_count: int


@dataclass(slots=True)
class CompletionDiagnostics:
    """Reference-aware statistics for one sampled completion."""

    actor_mean_logprob: float
    old_policy_mean_logprob: float
    reference_mean_logprob: float
    approx_kl: float


@dataclass(slots=True)
class PolicyUpdateStats:
    """Summary statistics for one policy update."""

    loss: float
    mean_ratio: float
    clip_fraction: float


class NativePolicy:
    """Small wrapper around a causal LM for sampling and log-prob computation."""

    def __init__(self, model, old_policy_model, reference_model, tokenizer, optimizer, device: str) -> None:
        self.model = model
        self.old_policy_model = old_policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str,
        reference_model_name: str | None,
        learning_rate: float,
        weight_decay: float,
        device: str,
        seed: int,
    ) -> "NativePolicy":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "torch and transformers are required for the native training path."
            ) from exc

        torch.manual_seed(seed)
        resolved_device = _resolve_device(torch, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model.to(resolved_device)
        model.train()
        old_policy_model = copy.deepcopy(model)
        old_policy_model.to(resolved_device)
        old_policy_model.eval()
        for parameter in old_policy_model.parameters():
            parameter.requires_grad_(False)
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_name or model_name,
            trust_remote_code=True,
        )
        reference_model.to(resolved_device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return cls(
            model=model,
            old_policy_model=old_policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=resolved_device,
        )

    def sample_group(
        self,
        prompt_text: str,
        *,
        num_generations: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[CompletionSample]:
        """Sample multiple completions for one prompt."""

        import torch

        prompt_ids = self._encode_prompt(prompt_text)
        input_ids = prompt_ids.unsqueeze(0).repeat(num_generations, 1)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_token_count = int(prompt_ids.shape[0])
        samples: list[CompletionSample] = []
        for row in output_ids:
            full_ids = row.tolist()
            completion_ids = full_ids[prompt_token_count:]
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            samples.append(
                CompletionSample(
                    prompt_text=prompt_text,
                    completion_text=completion_text,
                    full_token_ids=full_ids,
                    prompt_token_count=prompt_token_count,
                    completion_token_count=len(completion_ids),
                )
            )
        return samples

    def sync_old_policy(self) -> None:
        """Freeze a snapshot of the actor before collecting data for the next update."""

        self.old_policy_model.load_state_dict(self.model.state_dict())
        self.old_policy_model.eval()
        for parameter in self.old_policy_model.parameters():
            parameter.requires_grad_(False)

    def score_samples(self, grouped_samples: list[list[CompletionSample]]) -> list[list[CompletionDiagnostics]]:
        """Compute actor/old/reference log-probs and approximate KL for sampled completions."""

        diagnostics: list[list[CompletionDiagnostics]] = []
        for samples in grouped_samples:
            group_stats: list[CompletionDiagnostics] = []
            for sample in samples:
                actor_logprob = self._mean_completion_logprob(sample, model=self.model, track_grad=False)
                old_policy_logprob = self._mean_completion_logprob(
                    sample,
                    model=self.old_policy_model,
                    track_grad=False,
                )
                reference_logprob = self._mean_completion_logprob(
                    sample,
                    model=self.reference_model,
                    track_grad=False,
                )
                group_stats.append(
                    CompletionDiagnostics(
                        actor_mean_logprob=actor_logprob,
                        old_policy_mean_logprob=old_policy_logprob,
                        reference_mean_logprob=reference_logprob,
                        approx_kl=max(0.0, actor_logprob - reference_logprob),
                    )
                )
            diagnostics.append(group_stats)
        return diagnostics

    def update_step(
        self,
        grouped_samples: list[list[CompletionSample]],
        grouped_advantages: list[list[float]],
        grouped_diagnostics: list[list[CompletionDiagnostics]],
        *,
        clip_range: float,
    ) -> PolicyUpdateStats:
        """Apply a PPO-style ratio objective using an old-policy snapshot."""

        import torch

        loss_terms: list[torch.Tensor] = []
        ratios: list[float] = []
        clipped_count = 0
        self.optimizer.zero_grad()

        for samples, advantages, diagnostics in zip(grouped_samples, grouped_advantages, grouped_diagnostics):
            for sample, advantage, diagnostic in zip(samples, advantages, diagnostics):
                if sample.completion_token_count == 0:
                    continue

                current_logprob = self._mean_completion_logprob(
                    sample,
                    model=self.model,
                    track_grad=True,
                )
                old_logprob = torch.tensor(float(diagnostic.old_policy_mean_logprob), device=self.device)
                ratio = torch.exp(current_logprob - old_logprob)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                advantage_tensor = torch.tensor(float(advantage), device=self.device)
                surrogate_unclipped = ratio * advantage_tensor
                surrogate_clipped = clipped_ratio * advantage_tensor
                loss_terms.append(-torch.minimum(surrogate_unclipped, surrogate_clipped))
                ratio_value = float(ratio.detach().cpu().item())
                ratios.append(ratio_value)
                if abs(ratio_value - float(clipped_ratio.detach().cpu().item())) > 1e-6:
                    clipped_count += 1

        if not loss_terms:
            return PolicyUpdateStats(loss=0.0, mean_ratio=1.0, clip_fraction=0.0)

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        self.optimizer.step()
        return PolicyUpdateStats(
            loss=float(loss.detach().cpu().item()),
            mean_ratio=sum(ratios) / len(ratios) if ratios else 1.0,
            clip_fraction=clipped_count / len(ratios) if ratios else 0.0,
        )

    def greedy_completion(self, prompt_text: str, *, max_new_tokens: int) -> str:
        """Generate one deterministic completion used for evaluation."""

        import torch

        prompt_ids = self._encode_prompt(prompt_text)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0, prompt_ids.shape[0] :].tolist()
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def save(self, output_dir: str | Path) -> None:
        """Save model and tokenizer."""

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _encode_prompt(self, prompt_text: str):
        formatted_prompt = _format_prompt(self.tokenizer, prompt_text)
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        return encoded["input_ids"][0].to(self.device)

    def _mean_completion_logprob(self, sample: CompletionSample, *, model, track_grad: bool):
        import torch
        import torch.nn.functional as F

        input_ids = torch.tensor([sample.full_token_ids], dtype=torch.long, device=self.device)

        if track_grad:
            outputs = model(input_ids=input_ids)
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids)

        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        completion_start = sample.prompt_token_count - 1
        completion_log_probs = token_log_probs[:, completion_start:]
        if completion_log_probs.numel() == 0:
            zero = torch.tensor(0.0, device=self.device)
            return zero if track_grad else 0.0
        mean_log_prob = completion_log_probs.mean()
        if track_grad:
            return mean_log_prob
        return float(mean_log_prob.detach().cpu().item())


def _format_prompt(tokenizer, prompt_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt_text
    return prompt_text


def _resolve_device(torch_module: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch_module.cuda.is_available() else "cpu"
