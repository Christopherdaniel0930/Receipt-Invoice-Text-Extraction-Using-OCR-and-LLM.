from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, error_code: str, message: str):
        self.status = status
        self.error_code = error_code
        self.message = message


def _body(status: int, error_code: str, message: str) -> dict:
    return {"status": status, "error_code": error_code, "message": message}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=_body(exc.status, exc.error_code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content=_body(422, "VALIDATION_ERROR", "Invalid request."))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content=_body(500, "INTERNAL_ERROR", "An internal error occurred."))
