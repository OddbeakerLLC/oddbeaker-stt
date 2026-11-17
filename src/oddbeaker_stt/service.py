"""HTTP service + CLI.

Library boundary: everything above this module is importable without FastAPI.
The service is a thin adapter for third-party apps that prefer HTTP.
"""

import argparse
import json
import sys

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings


def create_app(settings: Settings | None = None):
    from .core import transcribe_file

    settings = settings or Settings()
    app = FastAPI(title="oddbeaker-stt", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        data = await file.read()
        try:
            result = transcribe_file(data, settings)
        except Exception as exc:  # noqa: BLE001 - surface decode/backend errors
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return result.to_dict()

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oddbeaker-stt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("transcribe", help="transcribe an audio file")
    p_t.add_argument("path", help="audio file (wav/flac/mp3/ogg, any soundfile format)")
    p_t.add_argument("--json", action="store_true", help="full JSON output")

    p_s = sub.add_parser("serve", help="run the HTTP service")
    p_s.add_argument("--host", default=None)
    p_s.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "transcribe":
        from .core import transcribe_file

        result = transcribe_file(args.path)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"verbatim: {result.verbatim}")
            print(f"cleaned:  {result.cleaned}")
        return 0

    if args.cmd == "serve":
        import uvicorn

        settings = Settings().with_overrides(host=args.host, port=args.port)
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())