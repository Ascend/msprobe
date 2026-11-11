import re

ROLE_PRODUCER = "P"
ROLE_CONSUMER = "D"
ROLE_MIXED = "mixed"
REQUEST_INDEX_UNKNOWN = -1  # TokenRow.request_index：本 rank 无块表行号
PAD_SLOT_ID = -1  # 引擎用该值标记「本 rank 不写这个位置」（PCP/DCP 分片）

RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)
