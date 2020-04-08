#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file RLEvaluate.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-03 15:14

import time
import torch
import numpy as np
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.synchronize import drain_queue

from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import Monitor


class MinibatchRlEvalOnce(MinibatchRlEval):
    def train(self):
        env = self.sampler.EnvCls(**self.sampler.env_kwargs)
        env = TimeLimit(env, max_episode_steps=10000)
        env = Monitor(env, '/cache/monitor', video_callable=lambda episode_id: True, force=True)
        self.agent.initialize(env.spaces)
        self.agent.to_device(self.affinity.get("cuda_idx", None))

        reward_sum = 0.
        observation = env.reset()
        env.render()
        while True:
            observation = torch.tensor(observation, dtype=torch.float32)
            action = self.agent.step(observation, torch.ones((2,2)), None) # TODO
            action = np.array(action.action)
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            env.render()
            if done:
                break
        env.close()
        print('reward:', reward_sum)


class AsyncRlEvalOnce(AsyncRlEval):
    def train(self):
        raise NotImplementedError('Async Evaluate')
        self.startup()
        if self._eval:
            while self.ctrl.sampler_itr.value < 1:
                time.sleep(0.05)
            traj_infos = drain_queue(self.traj_infos_queue, n_sentinel=1)
            self.store_diagnostics(0, 0, traj_infos, ())
            self.log_diagnostics(0, 0, 0)
