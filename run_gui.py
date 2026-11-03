"""Start de Streamlit-GUI (werkt als script en als gebundelde Windows-exe)."""

from __future__ import annotations

import os
import socket
import sys
import webbrowser
from pathlib import Path
from threading import Timer


def _bundle_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _work_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    os.chdir(_work_dir())
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    app_py = _bundle_dir() / "app.py"
    if not app_py.is_file():
        raise FileNotFoundError(f"app.py niet gevonden: {app_py}")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print("Koelkastanalyse EN 62552-3")
    print(f"Werkt in: {os.getcwd()}")
    print(f"Open in de browser: {url}")
    print("Sluit dit venster om het programma te stoppen.")
    Timer(1.8, lambda: webbrowser.open(url)).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_py),
        f"--server.port={port}",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--browser.serverAddress=127.0.0.1",
        "--server.address=127.0.0.1",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
