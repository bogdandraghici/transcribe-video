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
        self.assertEqual(ranked[1], ("SPEAKER_01", 2))
        self.assertEqual(len(ranked), 2)

    def test_winner_no_coverage(self):
        winner, conf, ranked = dr.winner_for_span(0, 10, [])
        self.assertIsNone(winner)
        self.assertEqual(conf, 0.0)
        self.assertEqual(ranked, [])


class TestAssignSpans(unittest.TestCase):
    def test_spans_chain_to_next_timestamp(self):
        body = "[00:00:00] Speaker 1: a\n[00:00:10] Speaker 2: b"
        lines = dr.parse_lines(body)
        spans = dr.assign_spans(lines, media_duration=25.0)
        self.assertEqual(spans, [(0, 0.0, 10.0), (1, 10.0, 25.0)])

    def test_continuation_line_makes_no_span(self):
        body = "[00:00:00] Speaker 1: a\nfara timestamp\n[00:00:10] Speaker 2: b"
        lines = dr.parse_lines(body)
        spans = dr.assign_spans(lines, media_duration=20.0)
        self.assertEqual(spans, [(0, 0.0, 10.0), (2, 10.0, 20.0)])

    def test_same_second_turns_get_minimum_span(self):
        body = "[00:00:05] Speaker 1: a\n[00:00:05] Speaker 2: b"
        lines = dr.parse_lines(body)
        spans = dr.assign_spans(lines, media_duration=30.0)
        self.assertEqual(spans[0], (0, 5.0, 6.0))


if __name__ == "__main__":
    unittest.main()
