from pathlib import Path
import unittest

from nano_verl.prompt_source import load_prompts


class PromptSourceTests(unittest.TestCase):
    def test_load_prompts_reads_jsonl(self) -> None:
        tmp_path = Path("tests/.tmp_prompt_source")
        tmp_path.mkdir(exist_ok=True)
        prompts_path = tmp_path / "prompts.jsonl"
        self.addCleanup(lambda: tmp_path.rmdir())
        self.addCleanup(lambda: prompts_path.unlink(missing_ok=True))
        prompts_path.write_text(
            '\n'.join(
                [
                    '{"id":"p1","prompt":"What is 1+1?","task_type":"math","reference_answer":"2"}',
                    '{"id":"p2","prompt":"Name a color","task_type":"qa","reference_answer":"blue","keywords":["blue"]}',
                ]
            ),
            encoding="utf-8",
        )

        prompts = load_prompts(prompts_path)

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].prompt_id, "p1")
        self.assertEqual(prompts[1].metadata["keywords"], ["blue"])
