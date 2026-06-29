import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import diarize


class TestDiarizeHelpers(unittest.TestCase):
    def test_token_prefers_cli(self):
        self.assertEqual(diarize._resolve_token("cli-tok"), "cli-tok")

    def test_token_from_env(self):
        old = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "env-tok"
        try:
            self.assertEqual(diarize._resolve_token(None), "env-tok")
        finally:
            if old is None:
                del os.environ["HF_TOKEN"]
            else:
                os.environ["HF_TOKEN"] = old

    def test_diar_kwargs_exact_wins(self):
        self.assertEqual(diarize._diar_kwargs(3, 2, 5), {"num_speakers": 3})

    def test_diar_kwargs_bounds(self):
        self.assertEqual(diarize._diar_kwargs(None, 2, 5),
                         {"min_speakers": 2, "max_speakers": 5})

    def test_diar_kwargs_empty(self):
        self.assertEqual(diarize._diar_kwargs(None, None, None), {})


if __name__ == "__main__":
    unittest.main()
