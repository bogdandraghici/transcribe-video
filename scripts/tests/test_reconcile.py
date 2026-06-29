import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import diarize_reconcile as dr


class TestParseLines(unittest.TestCase):
    def test_parses_timestamped_turn(self):
        [ln] = dr.parse_lines("[00:01:05] Speaker 2: salut, ce faci")
        self.assertEqual(ln.ts, 65.0)
        self.assertEqual(ln.label, "Speaker 2")
        self.assertEqual(ln.label_num, 2)
        self.assertIsNone(ln.gender)
        self.assertEqual(ln.text, "salut, ce faci")

    def test_parses_gender_tag(self):
        [ln] = dr.parse_lines("[00:00:04] Speaker 1 (f): hello")
        self.assertEqual(ln.gender, "f")
        self.assertEqual(ln.label_num, 1)
        self.assertEqual(ln.text, "hello")

    def test_non_turn_lines_are_passthrough(self):
        lines = dr.parse_lines("# header\n\nplain continuation text")
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(l.ts is None and l.label is None for l in lines))
        self.assertEqual(lines[2].raw, "plain continuation text")

    def test_fmt_ts(self):
        self.assertEqual(dr._fmt_ts(65), "[00:01:05]")
        self.assertEqual(dr._fmt_ts(3661), "[01:01:01]")


if __name__ == "__main__":
    unittest.main()
