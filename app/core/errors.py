from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class PrismError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(PrismError):
    status_code = 400
    code = "invalid_request"


class EmptyFileError(InvalidRequestError):
    code = "empty_file"


class DocumentNotFoundError(PrismError):
    status_code = 404
    code = "document_not_found"


class FileTooLargeError(PrismError):
    status_code = 413
    code = "file_too_large"


class UnsupportedMediaTypeError(PrismError):
    status_code = 415
    code = "unsupported_media_type"


class InvalidPDFError(InvalidRequestError):
    code = "invalid_pdf"


class UnreadablePDFError(PrismError):
    status_code = 422
    code = "unreadable_pdf"


class EncryptedPDFError(PrismError):
    status_code = 422
    code = "encrypted_pdf"


class EmbeddingFailedError(PrismError):
    status_code = 500
    code = "embedding_failed"


def error_body(exc: PrismError) -> dict:
    return {"error": {"code": exc.code, "message": exc.message}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PrismError)
    async def _prism_error_handler(request: Request, exc: PrismError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_body(exc))

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An unexpected error occurred."}},
        )
