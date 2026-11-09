# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller-spec voor een zelfstandige Streamlit-GUI zonder Python-installatie."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "app.py"), "."),
    (str(ROOT / "main.py"), "."),
    (str(ROOT / "en62552.py"), "."),
    (str(ROOT / "run_gui.py"), "."),
]
binaries = []
hiddenimports = [
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.runtime.caching",
    "streamlit.web.cli",
    "streamlit.web.bootstrap",
    "matplotlib.backends.backend_agg",
    "plotly",
    "plotly.graph_objects",
    "plotly.subplots",
]

for pkg in (
    "streamlit",
    "altair",
    "polars",
    "matplotlib",
    "numpy",
    "PIL",
    "plotly",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in (
    "streamlit",
    "altair",
    "pandas",
    "polars",
    "matplotlib",
    "numpy",
    "jsonschema",
    "narwhals",
    "packaging",
    "click",
    "tornado",
    "blinker",
    "tenacity",
    "toml",
    "protobuf",
    "pillow",
    "gitpython",
    "watchdog",
    "plotly",
    "cachetools",
    "typing_extensions",
    "rich",
    "pydeck",
    "requests",
    "charset_normalizer",
    "idna",
    "urllib3",
    "certifi",
    "packaging",
    "smmap",
    "gitdb",
    "attrs",
    "referencing",
    "rpds.rpds",
    "jsonschema_specifications",
):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

a = Analysis(
    [str(ROOT / "run_gui.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        "pandas.tests",
        "numpy.tests",
        "matplotlib.tests",
        "polars.tests",
        "plotly.matplotlylib",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Koelkastanalyse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Koelkastanalyse",
)
