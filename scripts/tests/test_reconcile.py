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


class TestOverlap(unittest.TestCase):
    def test_overlap_basic(self):
        self.assertEqual(dr.overlap(0, 10, 5, 15), 5)
        self.assertEqual(dr.overlap(0, 10, 10, 20), 0)
        self.assertEqual(dr.overlap(0, 10, 20, 30), 0)

    def test_winner_picks_dominant_speaker(self):
        segs = [
            {"start": 0, "end": 8, "speaker": "SPEAKER_00"},
            {"start": 8, "end": 10, "speaker": "SPEAKER_01"},
        ]
        winner, conf, ranked = dr.winner_for_span(0, 10, segs)
        self.assertEqual(winner, "SPEAKER_00")
        self.assertAlmostEqual(conf, 0.8)
        self.assertEqual(ranked[0], ("SPEAKER_00", 8))

    def test_winner_no_coverage(self):
        winner, conf, ranked = dr.winner_for_span(0, 10, [])
        self.assertIsNone(winner)
        self.assertEqual(conf, 0.0)
        self.assertEqual(ranked, [])


if __name__ == "__main__":
    unittest.main()
