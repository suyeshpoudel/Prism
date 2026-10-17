from dataclasses import dataclass

from app.schemas.documents import Chunk, PageExtraction

_PARAGRAPH_BREAK = "\n\n"
_SENTENCE_ENDINGS = ".?!"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class TextSlice:
    """A contiguous slice of a page's text, with source offsets."""

    text: str
    start: int
    end: int


def _best_break(text: str, start: int, hard_end: int) -> int:
    """Choose a cut position in [start, hard_end].

    Prefers, in order: paragraph breaks, line breaks, sentence endings,
    whitespace — falling back to the hard limit for unbreakable runs.
    Never cuts below half the requested window so chunks stay substantive.
    """
    min_end = start + max((hard_end - start) // 2, 1)
    if hard_end >= len(text):
        return hard_end

    position = text.rfind(_PARAGRAPH_BREAK, min_end - 2, hard_end)
    if position != -1:
        return position + 2

    position = text.rfind("\n", min_end - 1, hard_end)
    if position != -1:
        return position + 1

    for index in range(hard_end - 1, min_end - 2, -1):
        if text[index] in _SENTENCE_ENDINGS and text[index + 1].isspace():
            return index + 1

    position = text.rfind(" ", min_end - 1, hard_end)
    if position != -1:
        return position

    return hard_end


def chunk_text(text: str, config: ChunkingConfig) -> list[TextSlice]:
    """Split one page's text into overlapping slices with source offsets.

    Chunks are exact slices of ``text`` (no trimming), so
    ``text[slice.start:slice.end] == slice.text`` always holds. Whitespace-only
    slices are skipped. Chunks never span pages.
    """
    slices: list[TextSlice] = []
    cursor = 0
    length = len(text)

    while cursor < length:
        hard_end = min(cursor + config.chunk_size, length)
        end = hard_end if hard_end >= length else _best_break(text, cursor, hard_end)
        if end <= cursor:
            end = hard_end

        chunk = text[cursor:end]
        if chunk.strip():
            slices.append(TextSlice(text=chunk, start=cursor, end=end))

        if end >= length:
            break
        cursor = max(end - config.chunk_overlap, cursor + 1)

    return slices


def chunk_pages(
    *, document_id: str, pages: list[PageExtraction], config: ChunkingConfig
) -> list[Chunk]:
    """Chunk every page of a document, preserving page numbers and offsets.

    Chunk ids are deterministic: ``{document_id}:{chunk_index}``. ``chunk_index``
    runs sequentially across the whole document in page order. Blank pages
    contribute no chunks, but later pages keep their true page numbers.
    """
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in pages:
        for text_slice in chunk_text(page.text, config):
            chunks.append(
                Chunk(
                    id=f"{document_id}:{chunk_index}",
                    document_id=document_id,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text_slice.text,
                    char_count=len(text_slice.text),
                    start_offset=text_slice.start,
                    end_offset=text_slice.end,
                )
            )
            chunk_index += 1
    return chunks
