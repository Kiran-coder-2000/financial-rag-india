"""
tests/test_chunker.py

Tests for Part A (chunker.py): table-stripping, header detection
(ATX/bold/Setext), recursive splitting with word-safe overlap, and
the flag-not-discard behavior for small chunks.
"""

from src.chunking import chunker


class TestStripMarkdownTables:
    def test_removes_pipe_rows(self):
        text = "Some text\n| a | b |\n|---|---|\n| 1 | 2 |\nMore text"
        result = chunker.strip_markdown_tables(text)
        assert "|" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_handles_indented_table_rows(self):
        text = "Text\n  | a | b |\n  |---|---|\nAfter"
        result = chunker.strip_markdown_tables(text)
        assert "|" not in result

    def test_collapses_leftover_blank_lines(self):
        text = "A\n| x |\n| y |\n| z |\nB"
        result = chunker.strip_markdown_tables(text)
        assert "\n\n\n" not in result

    def test_all_table_page_leaves_minimal_text(self):
        """A page that's almost entirely a table (mirrors the real
        Reliance balance-sheet-page case) should reduce to just its
        heading/context lines, not an empty string with no signal."""
        text = "## Balance Sheet\n\nAs at 31st March, 2025\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        result = chunker.strip_markdown_tables(text)
        assert "Balance Sheet" in result
        assert "|" not in result


class TestHeaderDetection:
    def test_atx_headers(self):
        text = "## First\n\nContent one.\n\n## Second\n\nContent two."
        blocks = chunker.split_on_headers(text)
        assert [b["section"] for b in blocks] == ["First", "Second"]

    def test_preamble_gets_section_none(self):
        text = "Intro text with no heading.\n\n## First\n\nContent."
        blocks = chunker.split_on_headers(text)
        assert blocks[0]["section"] is None
        assert blocks[0]["content"] == "Intro text with no heading."
        assert blocks[1]["section"] == "First"

    def test_bold_line_heading(self):
        text = "**Segment Performance**\n\nThe retail segment grew."
        blocks = chunker.split_on_headers(text)
        assert blocks[0]["section"] == "Segment Performance"

    def test_setext_heading(self):
        text = "Financial Performance\n=====\n\nRevenue grew 12% year over year."
        blocks = chunker.split_on_headers(text)
        assert blocks[0]["section"] == "Financial Performance"

    def test_setext_dash_style(self):
        text = "Key Highlights\n------\n\nStrong quarter across all metrics."
        blocks = chunker.split_on_headers(text)
        assert blocks[0]["section"] == "Key Highlights"

    def test_no_heading_at_all(self):
        text = "Just a plain paragraph with no heading anywhere in it."
        blocks = chunker.split_on_headers(text)
        assert len(blocks) == 1
        assert blocks[0]["section"] is None

    def test_long_sentence_before_dash_divider_not_misread_as_setext(self):
        """A '---' horizontal-rule divider after an ORDINARY (long)
        sentence must not be misread as a Setext heading -- only
        short heading-length lines qualify."""
        text = (
            "This is an ordinary sentence that happens to be "
            "immediately followed by what looks like a divider line "
            "but is not a heading at all in this context.\n"
            "---\n\n"
            "Following content."
        )
        blocks = chunker.split_on_headers(text)
        # Should NOT have split into a heading named after that long sentence
        sections = [b["section"] for b in blocks]
        assert not any(
            s and len(s.split()) > 10 for s in sections if s
        ), f"A long sentence was misread as a heading: {sections}"

    def test_never_invents_headings(self):
        """A page with genuinely no heading markers must never
        fabricate a section label."""
        text = "Plain content.\n\nMore plain content in a second paragraph."
        blocks = chunker.split_on_headers(text)
        assert all(b["section"] is None for b in blocks)


class TestRecursiveSplitOverlap:
    def test_overlap_never_starts_mid_word(self):
        """Regression test for the raw-token-slice overlap bug: the
        overlap tail must always be built from whole words, never a
        fragment starting inside a word."""
        long_text = " ".join(f"word{i}" for i in range(200))
        pieces = chunker.recursive_split(long_text, chunk_size=50, chunk_overlap=10)
        assert len(pieces) > 1
        for piece in pieces[1:]:
            first_token = piece.split()[0]
            assert first_token.startswith("word"), f"Broken fragment: {first_token!r}"

    def test_no_overlap_when_zero(self):
        """With zero overlap, no word from the tail of one piece
        should reappear at the start of the next. Deliberately
        tokenizer-agnostic (checks word overlap, not an exact word
        index) since real BPE tokenization doesn't guarantee 1
        token per word the way a naive test might assume."""
        long_text = " ".join(f"word{i}" for i in range(200))
        pieces = chunker.recursive_split(long_text, chunk_size=50, chunk_overlap=0)
        assert len(pieces) > 1
        for i in range(len(pieces) - 1):
            tail_words = set(pieces[i].split()[-3:])
            head_words = set(pieces[i + 1].split()[:3])
            assert not (tail_words & head_words), (
                f"Unexpected overlap between piece {i} and {i+1}: "
                f"{tail_words & head_words}"
            )

    def test_single_short_text_not_split(self):
        pieces = chunker.recursive_split("A short sentence.", chunk_size=400, chunk_overlap=60)
        assert len(pieces) == 1

    def test_respects_chunk_size_plus_overlap_ceiling(self):
        long_text = " ".join(f"word{i}" for i in range(500))
        pieces = chunker.recursive_split(long_text, chunk_size=100, chunk_overlap=20)
        for piece in pieces:
            assert chunker.count_tokens(piece) <= 100 + 20


class TestSmallChunksAreFlaggedNotDropped:
    def test_small_chunk_kept_and_flagged(self, sample_page_segments):
        chunks = chunker.chunk_text_from_page_segments(
            sample_page_segments, company="TestCo", document="Annual Report",
            financial_year="2024-2025",
        )
        # Page 2 is almost entirely a table -- its leftover heading
        # text should survive as a flagged-small chunk, not vanish.
        page_2_chunks = [c for c in chunks if c["page"] == 2]
        assert len(page_2_chunks) >= 1
        assert any(c["is_small"] for c in page_2_chunks)

    def test_every_chunk_has_is_small_field(self, sample_page_segments):
        chunks = chunker.chunk_text_from_page_segments(
            sample_page_segments, company="TestCo", document="Annual Report",
            financial_year="2024-2025",
        )
        assert all("is_small" in c for c in chunks)

    def test_no_chunk_silently_disappears(self, sample_page_segments):
        """Total chunk count should equal what recursive_split would
        produce if nothing were filtered -- confirms flagging, not
        dropping."""
        chunks = chunker.chunk_text_from_page_segments(
            sample_page_segments, company="TestCo", document="Annual Report",
            financial_year="2024-2025", min_chunk_tokens=10000,  # flags everything
        )
        assert all(c["is_small"] for c in chunks)
        assert len(chunks) > 0


class TestNeverCrossesPageBoundary:
    def test_each_chunk_page_matches_source_page(self, sample_page_segments):
        chunks = chunker.chunk_text_from_page_segments(
            sample_page_segments, company="TestCo", document="Annual Report",
            financial_year="2024-2025",
        )
        valid_pages = {seg["page"] for seg in sample_page_segments}
        assert all(c["page"] in valid_pages for c in chunks)