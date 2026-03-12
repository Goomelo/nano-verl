import unittest

from nano_verl.native.grpo import apply_kl_penalty, clipped_surrogate_objective, compute_group_advantages


class NativeGRPOTests(unittest.TestCase):
    def test_group_advantages_are_zero_mean_per_prompt(self) -> None:
        grouped = [[0.0, 1.0, 2.0], [5.0, 5.0]]

        advantages = compute_group_advantages(grouped)

        self.assertAlmostEqual(sum(advantages[0]) / len(advantages[0]), 0.0, places=6)
        self.assertEqual(advantages[1], [0.0, 0.0])

    def test_kl_penalty_shapes_rewards(self) -> None:
        shaped = apply_kl_penalty(
            [[1.0, 0.5]],
            [[0.2, 0.1]],
            kl_coef=0.5,
        )

        self.assertEqual(shaped, [[0.9, 0.45]])

    def test_clipped_surrogate_caps_large_positive_ratio(self) -> None:
        unclipped = 1.5 * 1.0
        clipped = clipped_surrogate_objective(1.5, 1.0, clip_range=0.2)

        self.assertGreater(unclipped, clipped)
        self.assertEqual(clipped, 1.2)
