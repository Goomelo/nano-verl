import unittest

from nano_verl.backends.factory import create_rollout_backend, create_training_backend


class BackendTests(unittest.TestCase):
    def test_create_mock_rollout_backend(self) -> None:
        backend = create_rollout_backend("mock")
        self.assertEqual(backend.backend_name, "mock")

    def test_vllm_backend_requires_base_url(self) -> None:
        with self.assertRaises(ValueError):
            create_rollout_backend("vllm-server")

    def test_megatron_training_backend_is_stub(self) -> None:
        backend = create_training_backend("megatron")
        plan = backend.plan("Qwen/Qwen3-8B", "outputs/demo")

        self.assertEqual(plan.backend_name, "megatron")
        self.assertIn("torchrun", plan.launch_example)
