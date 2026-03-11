import unittest
from types import SimpleNamespace

from nano_verl.trainer.config import GRPOExperimentConfig
from nano_verl.trainer.dataflow import build_dataflow_diagram, build_stage_traces, format_stage_traces


def _config() -> GRPOExperimentConfig:
    return GRPOExperimentConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        train_data="data/grpo_math_train.jsonl",
        eval_data="data/grpo_math_eval.jsonl",
        output_dir="outputs/demo",
        max_steps=10,
        save_steps=5,
        logging_steps=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=64,
        temperature=0.8,
        top_p=0.95,
        seed=7,
        beta=0.0,
        loss_type="dr_grpo",
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj"],
        load_in_4bit=False,
        bf16=False,
        gradient_checkpointing=True,
        generation_backend="transformers",
        vllm_server_base_url=None,
        vllm_server_host="127.0.0.1",
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.3,
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=None,
        vllm_model_impl="vllm",
        megatron_tensor_parallel_size=1,
        megatron_pipeline_parallel_size=1,
        enable_think_reward=False,
        report_to="none",
    )


class _DummyTokenizer:
    def __len__(self) -> int:
        return 1024


class DataflowTests(unittest.TestCase):
    def test_dataflow_diagram_mentions_rollout_backend(self) -> None:
        diagram = build_dataflow_diagram(_config())
        self.assertIn("transformers rollout", diagram)

    def test_stage_trace_contains_expected_sections(self) -> None:
        training_args = SimpleNamespace(
            max_steps=10,
            learning_rate=1e-5,
            num_generations=4,
            max_completion_length=64,
        )
        traces = build_stage_traces(
            _config(),
            train_dataset=[{"prompt": [{"role": "user", "content": "hi"}], "solution": "1", "task": "math"}],
            eval_dataset=None,
            processing_class=_DummyTokenizer(),
            peft_config=None,
            reward_funcs=[lambda *_args, **_kwargs: [1.0]],
            training_args=training_args,
        )

        rendered = format_stage_traces(traces)

        self.assertIn("load_dataset", rendered)
        self.assertIn("reward_funcs", rendered)
        self.assertIn("output_dir=outputs/demo", rendered)
