from __future__ import annotations

import sys
import re
import tempfile
import unittest
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import preprocess_rtf_to_paragraphs as prep  # noqa: E402
from language_resources import load_date_locale, load_geothermal_patterns  # noqa: E402


def sentence(word: str, n_words: int = 10) -> str:
    words = [word] * n_words
    return " ".join(words).capitalize() + "."


class ParagraphSplittingTests(unittest.TestCase):
    def test_german_date_locale_normalizes_weekday_month_and_day_dot(self) -> None:
        month_translations, weekday_names = load_date_locale(PROJECT_DIR, "german")

        cleaned = prep.cleanup_date_text(
            "Dienstag, 7. März 2020",
            month_translations,
            weekday_names,
        )

        self.assertEqual(cleaned, "7 March 2020")

    def test_german_header_date_parses_from_separate_rtf_shape(self) -> None:
        month_translations, weekday_names = load_date_locale(PROJECT_DIR, "german")
        article = """Strom aus der Tiefe
Frankfurter Neue Presse (Regionalausgaben)
Dienstag 7. Juli 2020
Rüsselsheimer Echo

Section: ECHO LOKALES; S. 6
Length: 75 words
Body

Strom aus der Tiefe. Tief unter der Erdoberfläche herrschen hohe Temperaturen.

Load-Date: July 6, 2020
"""

        meta = prep.parse_header_metadata(article, month_translations, weekday_names)

        self.assertEqual(meta["date"], "7 July 2020")
        self.assertEqual(meta["newspaper"], "Frankfurter Neue Presse (Regionalausgaben)")

    def test_german_geothermal_patterns_include_erdwaerme_variants(self) -> None:
        patterns = load_geothermal_patterns(PROJECT_DIR, "german")
        combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), flags=re.IGNORECASE)

        self.assertRegex("Erdwärme und Tiefengeothermie", combined)
        self.assertRegex("Erdwaerme und geothermische Wärme", combined)

    def test_cli_language_override_wins_over_config_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("language: german\n", encoding="utf-8")

            language = prep.resolve_workflow_language(config_path, "it")
            month_translations, _ = load_date_locale(PROJECT_DIR, language)
            patterns = load_geothermal_patterns(PROJECT_DIR, language)

            self.assertEqual(language, "italian")
            self.assertIn("gennaio", month_translations)
            self.assertTrue(any("geoterm" in pattern for pattern in patterns))

    def test_snakemake_preprocess_rules_pass_wildcard_language(self) -> None:
        snakefile = (PROJECT_DIR / "Snakefile").read_text(encoding="utf-8")

        self.assertGreaterEqual(snakefile.count("--language {wildcards.language}"), 2)

    def test_cleanup_removes_start_bylines_conservatively(self) -> None:
        body = "Door Jan Jansen\njan@example.org\nDit is een normale alinea. Die blijft staan."

        cleaned, stats = prep.clean_body_before_paragraph_split(body, "dutch")

        self.assertNotIn("Door Jan Jansen", cleaned)
        self.assertNotIn("jan@example.org", cleaned)
        self.assertIn("Dit is een normale alinea.", cleaned)
        self.assertEqual(stats.removed_bylines, 2)

        kept, kept_stats = prep.clean_body_before_paragraph_split(
            "Di recente il comune ha approvato il progetto.",
            "italian",
        )
        self.assertIn("Di recente", kept)
        self.assertEqual(kept_stats.removed_bylines, 0)

    def test_cleanup_removes_graphic_marker_and_immediate_caption(self) -> None:
        body = "Eerste echte alinea.\nGraphic\nFoto: Jan Jansen\nTweede echte alinea."

        cleaned, stats = prep.clean_body_before_paragraph_split(body, "german")

        self.assertNotIn("Graphic", cleaned)
        self.assertNotIn("Foto: Jan Jansen", cleaned)
        self.assertIn("Eerste echte alinea.", cleaned)
        self.assertIn("Tweede echte alinea.", cleaned)
        self.assertEqual(stats.removed_graphic_artifacts, 1)
        self.assertEqual(stats.removed_captions, 1)

    def test_cleanup_removes_only_duplicated_pull_quotes(self) -> None:
        quote = '"Warmte uit de aarde is belangrijk"'
        body = (
            f"{quote}\n"
            "Warmte uit de aarde is belangrijk voor deze buurt en komt terug in het project.\n"
            '"Een unieke losse quote blijft staan"'
        )

        cleaned, stats = prep.clean_body_before_paragraph_split(body, "dutch")

        self.assertNotIn(quote, cleaned)
        self.assertIn('"Een unieke losse quote blijft staan"', cleaned)
        self.assertEqual(stats.removed_pull_quotes, 1)

    def test_summary_output_is_four_quiet_lines(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            prep.print_summary(
                "dutch",
                {
                    "loaded_documents": 1,
                    "unique_documents": 2,
                    "documents_after_cleaning": 3,
                    "paragraphs": 4,
                },
            )

        self.assertEqual(
            buf.getvalue().splitlines(),
            [
                'dutch loaded documents: "1"',
                'dutch unique documents: "2"',
                'dutch documents after cleaning: "3"',
                'dutch paragraphs: "4"',
            ],
        )

    def test_raw_only_quiet_mode_does_not_print_zero_stage_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            config_path = project_dir / "config.yaml"
            input_dir = project_dir / "input"
            output_raw = project_dir / "raw.csv"
            input_dir.mkdir()
            config_path.write_text("language: italian\n", encoding="utf-8")

            argv = [
                "preprocess_rtf_to_paragraphs.py",
                "--project-dir",
                str(project_dir),
                "--config",
                str(config_path),
                "--input-rtf-dir",
                str(input_dir),
                "--output-raw-articles-csv",
                str(output_raw),
                "--raw-only",
            ]
            df_raw = prep.pd.DataFrame(
                [{"title": "T", "newspaper": "N", "date": "", "body": "Body"}],
                columns=prep.RAW_ARTICLE_COLUMNS,
            )

            buf = io.StringIO()
            cwd = Path.cwd()
            with patch.object(sys, "argv", argv), patch.object(prep, "load_all_rtf_articles", return_value=df_raw):
                try:
                    with redirect_stdout(buf):
                        prep.main()
                finally:
                    os.chdir(cwd)

            self.assertEqual(buf.getvalue(), "")
            self.assertTrue(output_raw.exists())

    def test_discover_rtf_paths_recurses_case_insensitively_and_skips_doclists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "german" / "Bestanden (25)"
            nested.mkdir(parents=True)
            article_upper = nested / "Strom aus der Tiefe.RTF"
            article_lower = nested / "separate_article.rtf"
            doclist = nested / "Bestanden (25)_doclist.RTF"
            non_rtf = nested / "notes.txt"

            for path in [article_upper, article_lower, doclist, non_rtf]:
                path.write_text("x", encoding="utf-8")

            discovered = [path.relative_to(root) for path in prep.discover_rtf_paths(root)]

            self.assertEqual(
                discovered,
                [
                    Path("german/Bestanden (25)/Strom aus der Tiefe.RTF"),
                    Path("german/Bestanden (25)/separate_article.rtf"),
                ],
            )

    def test_normalize_layout_text_merges_wraps_and_preserves_blank_hints(self) -> None:
        text = "Dit is aard-\nwarmte.\nNog een regel.\n\nTweede blok\nloopt door."

        normalized = prep.normalize_layout_text(text)

        self.assertEqual(
            normalized,
            "Dit is aardwarmte. Nog een regel.\n\nTweede blok loopt door.",
        )

    def test_group_sentences_respects_size_limits_where_possible(self) -> None:
        sentences = [sentence(f"zin{i}") for i in range(18)]

        paragraphs = prep.group_sentences_into_paragraphs(sentences)

        self.assertGreater(len(paragraphs), 1)
        for paragraph in paragraphs:
            words = len(paragraph.split())
            sentence_count = prep.estimate_sentence_count(paragraph)
            self.assertLessEqual(words, prep.MAX_WORDS_PER_PARAGRAPH)
            self.assertLessEqual(sentence_count, prep.MAX_SENTENCES_PER_PARAGRAPH)

    def test_group_sentences_merges_short_final_chunk_backward_when_it_fits(self) -> None:
        sentences = [sentence(f"zin{i}", 25) for i in range(6)]

        paragraphs = prep.group_sentences_into_paragraphs(sentences)

        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].count("."), 6)
        self.assertGreaterEqual(len(paragraphs[0].split()), prep.MIN_WORDS_PER_PARAGRAPH)

    def test_heading_carry_attaches_to_next_paragraph(self) -> None:
        body = (
            "Belangrijke tussenkop\n\n"
            + " ".join(sentence(f"zin{i}", 12) for i in range(4))
        )

        paragraphs = prep.split_paragraphs_content_based(body, "nl")

        self.assertEqual(len(paragraphs), 1)
        self.assertTrue(paragraphs[0].startswith("Belangrijke tussenkop "))
        self.assertGreaterEqual(len(paragraphs[0].split()), prep.MIN_WORDS_PER_PARAGRAPH)

    def test_trailing_heading_attaches_backward_only_as_fallback(self) -> None:
        body = (
            " ".join(sentence(f"zin{i}", 12) for i in range(4))
            + "\n\n"
            + "Losse tussenkop"
        )

        paragraphs = prep.split_paragraphs_content_based(body, "nl")

        self.assertEqual(len(paragraphs), 1)
        self.assertTrue(paragraphs[0].endswith("Losse tussenkop"))


if __name__ == "__main__":
    unittest.main()
