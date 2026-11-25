from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from msprobe.vllm_kv_guard import checker, model_runner, scheduler, worker
from msprobe.vllm_kv_guard.checker import KVGuardError

from test_model_runner import batch, runner


class TestSlotAuditRuntime(TestCase):
    def test_scheduler_to_async_result_with_real_guard_and_recorder(self):
        for mode in ("NONE", "FULL"):
            for action in ("log", "raise"):
                for collision in (False, True):
                    with self.subTest(mode=mode, action=action, collision=collision):
                        self._run_step(mode, action, collision)

    def _run_step(self, mode, action, collision):
        npu = NS(Event=Mock(return_value=Mock()), current_stream=Mock(return_value=object()))
        options = {"action": action, "slot": True}
        config = NS(speculative_config=None)
        blocks = [1, 1 if collision else 2]
        scheduled = NS(
            num_scheduled_tokens={"a": 2, "b": 2}, total_num_scheduled_tokens=4,
            finished_req_ids=set(), preempted_req_ids=set(),
            scheduled_new_reqs=[NS(req_id=req, block_ids=[[block]], num_computed_tokens=0, replay_start=None)
                                for req, block in zip(("a", "b"), blocks)],
            scheduled_cached_reqs=NS(req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()),
            new_block_ids_to_zero=[], kv_cache_block_copies=None, kv_connector_metadata=None,
        )
        core = NS(vllm_config=config, schedule=Mock(return_value=scheduled), update_from_output=Mock())
        core.kv_cache_manager = NS(free=Mock())
        native = runner()
        result = NS(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}, sampled_token_ids=[[1], [2]],
                    kv_connector_output=None)
        lazy = NS(get_output=Mock(return_value=result))
        original_get = lazy.get_output

        def execute(_scheduled):
            supplied = batch(req_ids=["a", "b"], num_reqs=2, num_tokens=4, num_tokens_after_padding=8,
                             query_start_loc=torch.tensor([0, 2, 4]), positions=torch.tensor([0, 1, 0, 1]),
                             seq_lens=torch.tensor([2, 2]))
            slots = torch.tensor([[value for block in blocks for value in (block * 2, block * 2 + 1)] + [-1] * 4])
            tables = [torch.tensor([[block] for block in blocks])]
            native.model_state.prepare_attn(supplied, NS(name=mode), tables, slots, None, None)
            # The audit must use its frozen snapshot, even if native buffers are subsequently reused.
            slots.fill_(-1)
            tables[0].fill_(0)
            return lazy

        native_worker = NS(rank=0, use_v2_model_runner=True, model_runner=native,
                           execute_model=Mock(side_effect=execute), sample_tokens=Mock())
        with patch.object(torch, "npu", npu, create=True), patch.object(checker, "emit") as emitted, \
                patch.object(model_runner, "emit"), patch.object(scheduler, "emit"), patch.object(worker, "emit"):
            scheduler._observe(core, options)
            worker._observe(native_worker, options)
            native.initialize_kv_cache()
            plan = core.schedule()
            self.assertEqual(plan._msprobe_kv_plan,
                             (("a", 0, 2, ((1,),)), ("b", 0, 2, ((blocks[1],),))))
            self.assertIs(native_worker.execute_model(plan), lazy)
            original_get.assert_not_called()
            if collision and action == "raise":
                with self.assertRaisesRegex(KVGuardError, "slot_collision"):
                    lazy.get_output()
                core.update_from_output.__wrapped__.assert_not_called()
                # A latched failure blocks the next native execution as well.
                with self.assertRaisesRegex(KVGuardError, "slot_collision"):
                    native_worker.execute_model(plan)
                self.assertEqual(native_worker.execute_model.__wrapped__.call_count, 1)
            else:
                self.assertIs(lazy.get_output(), result)
                core.update_from_output(plan, result)
                core.update_from_output.__wrapped__.assert_called_once_with(plan, result)
            reasons = [call.kwargs["reason"] for call in emitted.call_args_list]
            self.assertEqual(set(reasons), {"slot_collision"} if collision else set())
            if collision:
                self.assertEqual({call.kwargs["source"] for call in emitted.call_args_list},
                                 {"scheduler", "execution"})
