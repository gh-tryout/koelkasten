"""Start de Streamlit-GUI (werkt als script en als gebundelde Windows-exe)."""

from __future__ import annotations

import os
import platform
import socket
import sys
import time
import traceback
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


def _wants_flag(*names: str) -> bool:
    args = {a.strip().lower() for a in sys.argv[1:]}
    return any(name.lower() in args for name in names)


def _debug_enabled() -> bool:
    if _wants_flag("--debug", "-debug", "/debug"):
        return True
    flag = str(os.environ.get("KOELKAST_DEBUG") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _selftest_enabled() -> bool:
    return _wants_flag("--selftest", "-selftest", "/selftest", "--diagnose", "-diagnose")


def _pause(message: str = "Druk op Enter om dit venster te sluiten...") -> None:
    if str(os.environ.get("KOELKAST_NOPAUSE") or "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            print(message)
            time.sleep(12)
            return
        input(message)
    except Exception:
        time.sleep(25)


def _open_log(work: Path):
    candidates = [
        work / "koelkastanalyse.log",
        Path(os.environ.get("TEMP") or ".") / "koelkastanalyse.log",
        Path.cwd() / "koelkastanalyse.log",
    ]
    last_err: Exception | None = None
    for path in candidates:
        try:
            fh = path.open("a", encoding="utf-8", errors="replace")
            return path, fh
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Kan geen logbestand schrijven ({last_err})")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                pass
        return len(text or "")

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        orig = self._streams[0] if self._streams else None
        return bool(orig and hasattr(orig, "isatty") and orig.isatty())


def _setup_stdio(log_file) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
    sys.stdout = _Tee(sys.__stdout__ or sys.stdout, log_file)
    sys.stderr = _Tee(sys.__stderr__ or sys.stderr, log_file)


def _prepare_runtime(work: Path) -> Path:
    runtime = work / "_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "mpl").mkdir(exist_ok=True)
    (runtime / "tmp").mkdir(exist_ok=True)
    (runtime / "streamlit").mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(runtime / "mpl")
    os.environ["TMP"] = str(runtime / "tmp")
    os.environ["TEMP"] = str(runtime / "tmp")
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
    return runtime


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _log_header(work: Path, bundle: Path) -> None:
    print("=" * 64)
    print("Koelkastanalyse EN 62552-3 — startlog")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 64)
    print(f"exe           : {sys.executable}")
    print(f"argv          : {sys.argv}")
    print(f"frozen        : {bool(getattr(sys, 'frozen', False))}")
    print(f"_MEIPASS      : {getattr(sys, '_MEIPASS', '')}")
    print(f"work_dir      : {work}")
    print(f"bundle_dir    : {bundle}")
    print(f"cwd           : {os.getcwd()}")
    print(f"python        : {sys.version}")
    print(f"platform      : {platform.platform()}")
    print(f"machine       : {platform.machine()}")
    try:
        print(f"home          : {Path.home()}")
    except Exception as exc:
        print(f"home          : FOUT {exc}")
    print(f"USERPROFILE   : {os.environ.get('USERPROFILE', '')}")
    print(f"TEMP          : {os.environ.get('TEMP', '')}")


def _try_import(name: str) -> str:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "")
        extra = f" {version}" if version else ""
        return f"OK{extra}"
    except Exception as exc:
        return f"FOUT: {type(exc).__name__}: {exc}"


def run_selftest(work: Path, bundle: Path) -> int:
    print()
    print("--- zelftest ---")
    errors = 0

    internal = work / "_internal"
    if getattr(sys, "frozen", False):
        if internal.is_dir():
            print(f"_internal     : OK ({internal})")
        else:
            print(f"_internal     : ONTBREEKT ({internal})")
            print("               Kopieer de hele map Koelkastanalyse, niet alleen de exe.")
            errors += 1

    app_py = bundle / "app.py"
    if app_py.is_file():
        print(f"app.py        : OK ({app_py})")
    else:
        print(f"app.py        : ONTBREEKT ({app_py})")
        errors += 1

    probe = work / "_schrijftest.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print("schrijven     : OK naast de exe")
    except Exception as exc:
        print(f"schrijven     : FOUT naast de exe: {exc}")
        errors += 1

    try:
        port = _free_port()
        print(f"localhost     : OK (vrije poort {port})")
    except Exception as exc:
        print(f"localhost     : FOUT, kan niet binden op 127.0.0.1: {exc}")
        errors += 1

    for name in (
        "streamlit",
        "polars",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "tkinter",
    ):
        result = _try_import(name)
        print(f"import {name:<12}: {result}")
        if result.startswith("FOUT"):
            errors += 1

    print()
    if errors:
        print(f"Zelftest: {errors} probleem/problemen.")
        print("Stuur koelkastanalyse.log mee (staat in deze map of in %TEMP%).")
        return 1
    print("Zelftest: geen blokkades gevonden. Start daarna zonder --selftest.")
    return 0


def start_streamlit(work: Path, bundle: Path, debug: bool) -> int:
    app_py = bundle / "app.py"
    if not app_py.is_file():
        raise FileNotFoundError(
            f"app.py niet gevonden: {app_py}. "
            "Kopieer de hele map (inclusief _internal), niet alleen de exe."
        )

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print("Koelkastanalyse EN 62552-3")
    print(f"Werkt in: {os.getcwd()}")
    print(f"Open in de browser: {url}")
    print("Sluit dit venster om het programma te stoppen.")
    if debug:
        print("DEBUG aan: extra logging staat in koelkastanalyse.log")

    if not debug:
        Timer(1.8, lambda: webbrowser.open(url)).start()
    else:
        print("DEBUG: browser wordt niet automatisch geopend. Kopieer de URL hierboven.")

    print("Streamlit importeren...")
    from streamlit.web import cli as stcli

    print("Streamlit starten...")
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
        "--logger.level=debug" if debug else "--logger.level=info",
    ]
    return int(stcli.main() or 0)


def main() -> int:
    debug = _debug_enabled()
    selftest = _selftest_enabled()
    work = _work_dir()
    try:
        log_path, log_file = _open_log(work)
    except Exception as exc:
        print(f"Kan geen logbestand openen: {exc}")
        traceback.print_exc()
        _pause()
        return 1
    try:
        _setup_stdio(log_file)
        print()
        os.chdir(work)
        _prepare_runtime(work)
        bundle = _bundle_dir()
        _log_header(work, bundle)
        print(f"logbestand    : {log_path}")
        if selftest:
            code = run_selftest(work, bundle)
            _pause()
            return code
        try:
            return start_streamlit(work, bundle, debug)
        except SystemExit as exc:
            code = exc.code
            if code in (0, None):
                return 0
            print(f"Streamlit stopte met code {code}", file=sys.stderr)
            traceback.print_exc()
            _pause()
            return int(code) if isinstance(code, int) else 1
    except Exception:
        traceback.print_exc()
        print()
        print("De app is gestopt door een fout. Zie de traceback hierboven")
        print(f"en het logbestand: {log_path}")
        _pause()
        return 1
    finally:
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
