#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file RLEvaluate.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-02-03 15:14

import time
import numpy as np
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.synchronize import drain_queue
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer

from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import Monitor

from k12ai.common.log_message import MessageMetric as MM


MONITOR_DIR = '/cache/monitor'


class MinibatchRlEvaluate(MinibatchRlEval):
    def train(self):
        env = self.sampler.EnvCls(**self.sampler.env_kwargs)
        env = TimeLimit(env, max_episode_steps=10000)
        env = Monitor(env, MONITOR_DIR, video_callable=lambda episode_id: True, force=True)

        self.agent.initialize(env.spaces)
        self.agent.to_device(self.affinity.get("cuda_idx", None))

        reward_sum = 0.
        observation = buffer_from_example(env.reset(), 1)
        action = buffer_from_example(env.action_space.null_value(), 1)
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, np.zeros(1, dtype="float32")))

        self.agent.reset()
        while True:
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            o, r, d, env_info = env.step(action[0])
            reward_sum += r
            if d:
                break
        env.close()

        mp4file = '{}/{}.video.{}.video000000.mp4'.format(MONITOR_DIR, env.file_prefix, env.file_infix)
        MM().add_text('game', 'result', str(reward_sum)).send()
        MM().add_video('episode', 'video', mp4file).send()


class AsyncRlEvaluate(AsyncRlEval):
    def train(self):
        raise NotImplementedError('Async Evaluate')
        self.startup()
        if self._eval:
            while self.ctrl.sampler_itr.value < 1:
                time.sleep(0.05)
            traj_infos = drain_queue(self.traj_infos_queue, n_sentinel=1)
            self.store_diagnostics(0, 0, traj_infos, ())
            self.log_diagnostics(0, 0, 0)
