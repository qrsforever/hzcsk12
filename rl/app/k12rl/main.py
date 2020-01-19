#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-19 17:34

import argparse
import os
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

logger = logging.getLogger(__name__)

from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.cpu.parallel_sampler import CpuParallelSampler
from rlpyt.samplers.cpu.collectors import WaitResetCollector
from rlpyt.samplers.async_.async_cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.async_.collectors import DbCpuResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.runners.minibatch_rl_eval import MinibatchRlEval
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context


def _rl_train(out_dir, config):
    # '_k12.agent.eps_final_min': False,
    # '_k12.algo.optimcls': 'adam',
    # '_k12.dataset': 'pong',
    # '_k12.model.name': 'dqn',
    # '_k12.model.network': 'dqn',
    # '_k12.runner.cuda_device': 2,
    # '_k12.sampler.eval': False,
    # '_k12.sampler.mode': 'gpu',
    # '_k12.task': 'atari',
    # 'affinity.async_sample': False,
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--phase',
            default=None,
            type=str,
            dest='phase',
            help="phase")
    parser.add_argument(
            '--config_file',
            default='/cache',
            type=str,
            dest='config_file',
            help="config file")
    parser.add_argument(
            '--out_dir',
            default='/cache',
            type=str,
            dest='out_dir',
            help="log dir")
    args = parser.parse_args()

    with open(os.join(args.config_file), 'r') as f:
        config = json.load(f)

    try:
        if args.phase == 'train':
            _rl_train(args.out_dir, config)
        else:
            logger.error('phase not impl yet')
    except Exception as err:
        logger.error('{}'.format(err))
