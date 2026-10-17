import pytest

from app.processing.chunking import ChunkingConfig, chunk_pages, chunk_text
from app.schemas.documents import PageExtraction


def make_pages(*texts: str) -> list[PageExtraction]:
    pages = []
    for index, text in enumerate(texts, start=1):
        pages.append(
            PageExtraction(
                page_number=index, text=text, char_count=len(text), is_empty=not text.strip()
            )
        )
    return pages


def test_short_text_single_chunk_with_exact_offsets():
    text = "Short page."

    slices = chunk_text(text, ChunkingConfig(chunk_size=100, chunk_overlap=10))

    assert len(slices) == 1
    assert slices[0].text == text
    assert (slices[0].start, slices[0].end) == (0, len(text))


def test_whitespace_only_text_yields_no_chunks():
    assert chunk_text("   \n\t\n  ", ChunkingConfig(chunk_size=10, chunk_overlap=2)) == []


def test_no_empty_trailing_chunk_on_exact_multiple():
    text = "a" * 20

    slices = chunk_text(text, ChunkingConfig(chunk_size=10, chunk_overlap=0))

    assert [len(s.text) for s in slices] == [10, 10]
    assert all(s.text.strip() for s in slices)


def test_hard_cut_for_unbreakable_run():
    text = "x" * 25

    slices = chunk_text(text, ChunkingConfig(chunk_size=10, chunk_overlap=0))

    assert slices[0].text == "x" * 10
    assert slices[1].text == "x" * 10
    assert slices[2].text == "x" * 5


def test_chunks_are_exact_slices_of_source_text():
    text = "Sentence one is here. Sentence two follows. Sentence three ends it."
    config = ChunkingConfig(chunk_size=20, chunk_overlap=5)

    for text_slice in chunk_text(text, config):
        assert text[text_slice.start : text_slice.end] == text_slice.text


def test_overlap_between_consecutive_chunks():
    words = " ".join(f"word{i}" for i in range(40))
    config = ChunkingConfig(chunk_size=60, chunk_overlap=20)

    slices = chunk_text(words, config)

    assert len(slices) > 1
    for previous, current in zip(slices, slices[1:], strict=False):
        # The next chunk starts inside the previous chunk's range.
        assert previous.start < current.start < previous.end
        # The tail of the previous chunk is the head of the next chunk.
        overlap_start = current.start
        assert previous.text[overlap_start - previous.start :] == current.text[: previous.end - overlap_start]


def test_no_gaps_and_monotonic_offsets():
    text = " ".join(f"word{i}" for i in range(100))
    config = ChunkingConfig(chunk_size=80, chunk_overlap=15)

    slices = chunk_text(text, config)

    assert slices[0].start == 0
    assert slices[-1].end == len(text)
    for previous, current in zip(slices, slices[1:], strict=False):
        assert current.start == previous.end - config.chunk_overlap
        assert current.start > previous.start


def test_prefers_paragraph_breaks():
    paragraph_a = "Alpha " * 8  # 48 chars
    paragraph_b = "Bravo " * 8
    paragraph_c = "Charlie " * 8
    text = f"{paragraph_a.strip()}\n\n{paragraph_b.strip()}\n\n{paragraph_c.strip()}"

    slices = chunk_text(text, ChunkingConfig(chunk_size=50, chunk_overlap=0))

    assert len(slices) >= 2
    first = slices[0].text
    # The cut lands on the paragraph break, so the break trails the chunk
    # and the next chunk starts cleanly at the next paragraph.
    assert first.endswith("\n\n")
    assert first.count("\n\n") == 1
    assert slices[1].text.startswith("Bravo")


def test_prefers_sentence_boundaries_when_no_paragraphs():
    text = "First sentence here. Second sentence here. Third sentence here."

    slices = chunk_text(text, ChunkingConfig(chunk_size=40, chunk_overlap=0))

    assert len(slices) >= 2
    assert slices[0].text.endswith(".")


def test_chunk_config_validation():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0)
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=20)


def test_chunk_pages_preserves_document_and_page_metadata():
    pages = make_pages("Page one text " * 10, "Page two text " * 10)

    chunks = chunk_pages(
        document_id="doc1", pages=pages, config=ChunkingConfig(chunk_size=50, chunk_overlap=10)
    )

    assert chunks
    assert all(chunk.document_id == "doc1" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.id for chunk in chunks] == [f"doc1:{i}" for i in range(len(chunks))]
    page_numbers = {chunk.page_number for chunk in chunks}
    assert page_numbers == {1, 2}
    for chunk in chunks:
        page = pages[chunk.page_number - 1]
        assert page.text[chunk.start_offset : chunk.end_offset] == chunk.text
        assert chunk.char_count == len(chunk.text)


def test_blank_pages_produce_no_chunks_but_keep_page_numbers():
    pages = make_pages("Real content on page three.", "", "More content on page five.")

    chunks = chunk_pages(
        document_id="doc2", pages=pages, config=ChunkingConfig(chunk_size=500, chunk_overlap=50)
    )

    assert [chunk.page_number for chunk in chunks] == [1, 3]


def test_chunk_index_is_sequential_across_pages():
    pages = make_pages("a" * 30, "b" * 30)

    chunks = chunk_pages(
        document_id="doc3", pages=pages, config=ChunkingConfig(chunk_size=20, chunk_overlap=0)
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
