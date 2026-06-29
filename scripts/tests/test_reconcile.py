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


class TestClusterMaps(unittest.TestCase):
    def _setup(self):
        body = ("[00:00:00] Speaker 1: a\n"
                "[00:00:10] Speaker 1: b\n"
                "[00:00:20] Speaker 2: c")
        lines = dr.parse_lines(body)
        segs = [
            {"start": 0, "end": 10, "speaker": "SPEAKER_05"},   # turn @0  -> 05
            {"start": 10, "end": 20, "speaker": "SPEAKER_09"},  # turn @10 -> 09
            {"start": 20, "end": 40, "speaker": "SPEAKER_05"},  # turn @20 -> 05
        ]
        spans = dr.assign_spans(lines, 40.0)
        winners = dr._compute_winners(lines, spans, segs)
        return lines, spans, winners

    def test_home_cluster_is_majority(self):
        lines, spans, winners = self._setup()
        homes = dr.home_clusters(lines, winners)
        # Speaker 1 spoke at 0 (05) and 10 (09) -> tie broken by count/order -> 05 or 09;
        # Speaker 2 spoke at 20 (05).
        self.assertEqual(homes["Speaker 2"], "SPEAKER_05")
        self.assertIn(homes["Speaker 1"], {"SPEAKER_05", "SPEAKER_09"})

    def test_first_appearance_numbering(self):
        segs = [
            {"start": 0, "end": 10, "speaker": "SPEAKER_05"},
            {"start": 10, "end": 20, "speaker": "SPEAKER_09"},
            {"start": 20, "end": 40, "speaker": "SPEAKER_05"},
        ]
        cmap = dr.first_appearance_map(segs)
        # Ordered by first segment start: SPEAKER_05 (t=0) then SPEAKER_09 (t=10).
        self.assertEqual(cmap["SPEAKER_05"], 1)
        self.assertEqual(cmap["SPEAKER_09"], 2)

    def test_first_appearance_includes_non_winners(self):
        # SPEAKER_02 never wins a span but must still get a stable id.
        segs = [
            {"start": 0, "end": 9, "speaker": "SPEAKER_00"},
            {"start": 9, "end": 10, "speaker": "SPEAKER_02"},
        ]
        cmap = dr.first_appearance_map(segs)
        self.assertEqual(cmap, {"SPEAKER_00": 1, "SPEAKER_02": 2})


class TestReconcile(unittest.TestCase):
    def test_relabels_and_flags_reattribution(self):
        # Gemini stapled the 2nd turn to Speaker 1, but acoustically it's a 2nd voice.
        body = ("[00:00:00] Speaker 1: intrebare\n"
                "[00:00:10] Speaker 1: raspuns lung de la alta persoana")
        segs = [
            {"start": 0, "end": 10, "speaker": "SPEAKER_00"},
            {"start": 10, "end": 30, "speaker": "SPEAKER_01"},
        ]
        out = dr.reconcile(body, segs, media_duration=30.0).splitlines()
        self.assertEqual(out[0], "[00:00:00] Speaker 1: intrebare")
        self.assertTrue(out[1].startswith("[00:00:10] Speaker 2 ‹reattr gemini=S1 conf=1.00›:"))

    def test_preserves_gender_tag(self):
        body = "[00:00:00] Speaker 1 (f): salut"
        segs = [{"start": 0, "end": 5, "speaker": "SPEAKER_00"}]
        out = dr.reconcile(body, segs, 5.0)
        self.assertEqual(out, "[00:00:00] Speaker 1 (f): salut")

    def test_mixed_turn_flag(self):
        body = "[00:00:00] Speaker 1: doi vorbitori in acelasi rand"
        segs = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_01"},
        ]
        out = dr.reconcile(body, segs, 10.0)
        self.assertIn("‹mixed S1/S2›", out)

    def test_passthrough_and_no_coverage(self):
        body = "# header\n[00:00:00] Speaker 1: fara segmente"
        out = dr.reconcile(body, [], media_duration=10.0).splitlines()
        self.assertEqual(out[0], "# header")
        self.assertEqual(out[1], "[00:00:00] Speaker 1: fara segmente")


if __name__ == "__main__":
    unittest.main()
