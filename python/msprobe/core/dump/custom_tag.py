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

import threading


class CustomTagManager:
    """Manager of user-defined custom tags.

    Two storages are maintained:

    - a thread-local stack (``_tls``) holding the tags currently in effect.
      ``tag(...)`` pushes on enter of the ``with`` block and pops on exit,
      so the stack depth equals the current nesting depth of tag scopes.
      Data collected inside a tag scope resolves its tag from the stack top.

    - a shared sticky table (``_sticky_tags``) mapping entry names to tags.
      It is filled by the forward hooks so that data collected later in time
      (backward tensors, outputs of asynchronous distributed ops) can still
      resolve the tag by entry name after the ``with`` block has exited.
      The table is cleared on every ``step()`` boundary, aligned with the
      lifecycle of the api/module call counters.

    Nesting control:

    - a thread-local module-nesting counter (``depth``) is maintained per tag
      scope. On ``push`` the counter is reset to 0 and the previous value is
      snapshotted (restored on ``pop``, which also heals counter imbalance
      caused by exceptions). Module hooks bracket each dumped module call
      with ``enter_module_call``/``leave_module_call``.
    - ``get_current_tag`` returns the label only while the nesting depth is
      0, i.e. for modules and free APIs called directly inside the ``with``
      block. Entries nested inside a called module (its submodules and its
      internal APIs) stay untagged, so the tag acts as a call anchor instead
      of a dye. There is no recursive option: a tag dyeing the whole subtree
      would make the tagged boundary ambiguous for the tag-based comparison.
    """

    _tls = threading.local()
    _sticky_tags = {}
    _lock = threading.Lock()
    # One-way flag, flipped to True on the first push(). Lets the per-entry hot
    # path (DataCollector._attach_custom_tag) short-circuit with a lock-free
    # read while the tag feature has never been used in this process.
    # Deliberately NOT cleared by reset(): flipping it back would re-open a
    # race where an entry gets skipped while a tag scope is being opened.
    _feature_used = False

    @classmethod
    def push(cls, label):
        # Flip before touching the stack so the very first entry collected
        # inside this scope can never be missed by the hot-path short circuit.
        cls._feature_used = True
        stack = getattr(cls._tls, "stack", None)
        if stack is None:
            stack = cls._tls.stack = []
        # snapshot the outer nesting depth so pop() can restore it; nesting is
        # re-counted from 0 inside this scope.
        stack.append((label, getattr(cls._tls, "depth", 0)))
        cls._tls.depth = 0

    @classmethod
    def pop(cls):
        stack = getattr(cls._tls, "stack", None)
        if stack:
            _, depth_snapshot = stack.pop()
            # restore the snapshot directly: self-heals any depth imbalance
            # left behind by an exception inside the with block.
            cls._tls.depth = depth_snapshot

    @classmethod
    def get_current_tag(cls):
        stack = getattr(cls._tls, "stack", None)
        if not stack:
            return None
        if getattr(cls._tls, "depth", 0) > 0:
            return None  # inside a called module: not the direct-call layer
        return stack[-1][0]

    @classmethod
    def enter_module_call(cls):
        cls._tls.depth = getattr(cls._tls, "depth", 0) + 1

    @classmethod
    def leave_module_call(cls):
        cls._tls.depth = max(0, getattr(cls._tls, "depth", 0) - 1)

    @classmethod
    def record_tags(cls, names, label):
        with cls._lock:
            for name in names:
                cls._sticky_tags[name] = label

    @classmethod
    def get_sticky_tag(cls, name):
        with cls._lock:
            return cls._sticky_tags.get(name)

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._sticky_tags.clear()
