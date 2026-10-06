"""Entry point: python -m app.main  (or: ai serve)"""
from __future__ import annotations

import uvicorn

from .api import create_app
from .config import load_config


def main() -> None:
    config = load_config()
    if config.application.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {config.application.host}; the API has no authentication.")
    uvicorn.run(create_app(config), host=config.application.host, port=config.application.port,
                log_level=config.application.log_level.lower())


if __name__ == "__main__":
    main()
