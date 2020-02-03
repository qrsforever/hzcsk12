#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file RLEvaluate.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-03 15:14

import time
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.synchronize import drain_queue


class MinibatchRlEvalOnce(MinibatchRlEval):
    def train(self):
        self.startup()
        eval_traj_infos, eval_time = self.evaluate_agent(0)
        self.log_diagnostics(0, eval_traj_infos, eval_time)


class AsyncRlEvalOnce(AsyncRlEval):
    def train(self):
        self.startup()
        if self._eval:
            while self.ctrl.sampler_itr.value < 1:
                time.sleep(0.05)
            traj_infos = drain_queue(self.traj_infos_queue, n_sentinel=1)
            self.store_diagnostics(0, 0, traj_infos, ())
            self.log_diagnostics(0, 0, 0)
