"""Bound receipt request bodies before multipart parsing buffers them."""

import json


class UploadBodyLimitMiddleware:
    def __init__(self, app, *, max_body_bytes: int, path: str):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path = path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != self.path:
            await self.app(scope, receive, send)
            return

        async def too_large():
            payload = json.dumps({
                "status": 413,
                "error_code": "FILE_TOO_LARGE",
                "message": "Image must be 10 MB or smaller.",
            }).encode()
            await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})

        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > self.max_body_bytes:
                await too_large()
                return
        except ValueError:
            pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await too_large()


class _BodyTooLarge(Exception):
    pass
