import pymupdf


def make_pdf(pages: list[str] | None = None, *, encrypt: bool = False, password: str = "secret") -> bytes:
    """Build an in-memory PDF. An empty string creates a blank (text-free) page."""
    document = pymupdf.open()
    for text in (pages if pages is not None else ["Hello Prism"]):
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    if encrypt:
        return document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password)
    return document.tobytes()
