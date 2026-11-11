import sys

from msprobe.core.common.log import logger

_registered = False


def _ascend_config_initialized() -> bool:
    module = sys.modules.get("vllm_ascend.ascend_config")
    return module is not None and getattr(module, "_ASCEND_CONFIG", None) is not None


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    from msprobe.kv_monitor.hooks.installer import INSTALLER

    if not _ascend_config_initialized():
        INSTALLER.install()
        logger.debug("[msprobe.kv_monitor] hooks installed via vllm general_plugins entry point")
