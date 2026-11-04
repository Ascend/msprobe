# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
#  Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
#  MindStudio is licensed under Mulan PSL v2.
#  You can use this software according to the terms and conditions of the Mulan PSL v2.
#  You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import re

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.dump.custom_tag import CustomTagManager


def _validate_label(label):
    if not isinstance(label, str) or not re.fullmatch(Const.CUSTOM_TAG_PATTERN, label):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            "invalid custom tag '{}'. A tag must be a non-empty string of at most 128 "
            "characters, containing only letters, digits, '_' and '-'.".format(label),
        )


class CustomTag:
    """User-defined custom tag for dump data annotation.

    Supported usage::

        # scope usage: entries dumped inside the block carry the tag
        with tag("attn_block"):
            attn = self.attention(x)
            y = F.gelu(x)

    By default the tag is non-recursive: only modules/APIs called directly
    inside the block carry it, while entries of nested submodules stay
    untagged (the tag acts as a call anchor). Pass ``recursive=True`` to
    propagate the tag to every nested submodule entry as well.

    Tags recorded inside the forward pass are propagated to the matching
    backward entries automatically.
    """

    def __init__(self, label, recursive=False):
        _validate_label(label)
        CustomTagManager.validate_recursive(recursive)
        self.label = label
        self.recursive = recursive

    def __enter__(self):
        CustomTagManager.push(self.label, self.recursive)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        CustomTagManager.pop()
        return False


def tag(label, recursive=False):
    """Create a custom tag, usable as a context manager.

    Args:
        label (str): tag name attached to the dumped entries.
        recursive (bool): if False (default), only modules/APIs called directly
            inside the tag scope are tagged; nested submodules stay untagged.
            If True, the tag propagates to every nested submodule entry.
    """
    return CustomTag(label, recursive)
