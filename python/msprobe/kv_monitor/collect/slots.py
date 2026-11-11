from typing import NamedTuple

import numpy as np

from msprobe.kv_monitor.engine.adapter import RankSlotMapping


class Segment(NamedTuple):
    """本步一个请求要取值的一段：全局位置 + 对应槽位。"""

    request_id: str
    positions: list[int]
    slots: np.ndarray


def window_segments(
    mapping: RankSlotMapping,
    windows: dict[str, tuple[list[int], int]],
) -> list[Segment]:
    """把 decode 侧的历史窗口解算成 (positions, slots)；覆盖不到窗口的请求整段跳过。"""
    segments = []
    for request_id, entry in windows.items():
        if not entry:
            continue
        blocks, window = entry
        tokens = mapping.local_tokens(int(window), blocks)
        if tokens is None:
            continue
        segments.append(Segment(request_id, tokens.positions, tokens.slots))
    return segments
