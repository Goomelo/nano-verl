import unittest

from nano_verl.reward import RuleReward
from nano_verl.selector import select_samples
from nano_verl.types import PromptExample, RolloutCandidate, RolloutSample


class RewardAndSelectorTests(unittest.TestCase):
    def test_rule_reward_scores_math_correctness(self) -> None:
        prompt = PromptExample(
            prompt_id="math-1",
            prompt="What is 2 + 3?",
            task_type="math",
            reference_answer="5",
        )
        candidate = RolloutCandidate(
            sample_id="math-1-cand-00",
            prompt_id="math-1",
            text="The answer is 5.",
            token_count=4,
            latency_ms=10.0,
        )

        result = RuleReward().score(prompt, candidate)

        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.is_correct)

    def test_selector_best_of_n_picks_highest_reward(self) -> None:
        prompt = PromptExample(
            prompt_id="math-2",
            prompt="What is 4 + 4?",
            task_type="math",
            reference_answer="8",
        )
        candidates = [
            RolloutCandidate("c0", "math-2", "7", 1, 1.0),
            RolloutCandidate("c1", "math-2", "8", 1, 1.0),
        ]
        rollout = RolloutSample(prompt=prompt, candidates=candidates)
        reward_fn = RuleReward()
        reward_lookup = {candidate.sample_id: reward_fn.score(prompt, candidate) for candidate in candidates}

        selected = select_samples([rollout], reward_lookup, strategy="best_of_n")

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].candidate.sample_id, "c1")
        self.assertTrue(selected[0].accepted)
