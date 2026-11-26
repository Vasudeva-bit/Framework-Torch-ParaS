import os
from pathlib import Path


def _find_library() -> str:
    env_path = os.environ.get("PJRT_PARAS_LIBRARY_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    pkg_lib = Path(__file__).resolve().parent / "libpjrt_paras.so"
    if pkg_lib.is_file():
        return str(pkg_lib)

    raise FileNotFoundError(
        f"libpjrt_paras.so not found at {pkg_lib}. "
        "Build it with pjrt_plugin/scripts/build_pjrt.sh or set PJRT_PARAS_LIBRARY_PATH."
    )


def initialize() -> None:
    from jax._src import xla_bridge as xb

    plugin_path = _find_library()
    xb.register_plugin(
        "paras",
        priority=500,
        library_path=plugin_path,
        options=None,
    )
