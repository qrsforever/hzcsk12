#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-19 17:34

import argparse
import os
import sys
import traceback
import json
import logging
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

logger = logging.getLogger(__name__)

from pyhocon import ConfigFactory

# K12AI
from k12rl.utils.rpc_message import hzcsk12_error_message as _err_msg

# utils
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.utils.logging.context import logger_context

# sync
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler

# async
from rlpyt.samplers.async_.serial_sampler import AsyncSerialSampler
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.async_.alternating_sampler import AsyncAlternatingSampler

# collectors
from rlpyt.samplers.async_.collectors import (DbCpuResetCollector, DbCpuWaitResetCollector)
from rlpyt.samplers.async_.collectors import (DbGpuResetCollector, DbGpuWaitResetCollector)
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector, CpuWaitResetCollector)
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)

# runner
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval

# algo & agent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.algos.dqn.r2d1 import R2D1

# atari
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1Agent
from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1AlternatingAgent


def _rl_runner(task, async_, mode, netw, model, optim, reset_, config):
    if task == 'atari':
        Env = AtariEnv
        Traj = AtariTrajInfo

    if mode == 'serial':
        if async_:
            Sampler = AsyncSerialSampler
            Collector = DbCpuResetCollector if reset_ else DbCpuWaitResetCollector
        else:
            Sampler = SerialSampler
            Collector = CpuResetCollector if reset_ else CpuWaitResetCollector
    elif mode == 'cpu':
        if async_:
            Sampler = AsyncCpuSampler
            Collector = DbCpuResetCollector if reset_ else DbCpuWaitResetCollector
        else:
            Sampler = CpuSampler
            Collector = CpuResetCollector if reset_ else CpuWaitResetCollector
    elif mode == 'gpu':
        if async_:
            Sampler = AsyncGpuSampler
            Collector = DbGpuResetCollector if reset_ else DbGpuWaitResetCollector
        else:
            Sampler = GpuSampler
            Collector = GpuResetCollector if reset_ else GpuWaitResetCollector
    elif mode == 'alternating':
        if async_:
            Sampler = AsyncAlternatingSampler
            Collector = DbGpuResetCollector if reset_ else DbGpuWaitResetCollector
        else:
            Sampler = AlternatingSampler
            Collector = GpuResetCollector if reset_ else GpuWaitResetCollector

    if async_:
        Runner = AsyncRlEval
    else:
        Runner = MinibatchRlEval

    if netw == 'dqn':
        if model == 'dqn':
            Algo = DQN
            if task == 'atari':
                Agent = AtariDqnAgent
        elif model == 'catdqn':
            Algo = CategoricalDQN
            if task == 'atari':
                Agent = AtariCatDqnAgent
        elif model == 'r2d1':
            Algo = R2D1
            if task == 'atari':
                if mode == 'alternating':
                    Agent = AtariR2d1AlternatingAgent
                else:
                    Agent = AtariR2d1Agent

    affinity = make_affinity(**config['affinity'])
    sampler = Sampler(
            EnvCls=Env,
            env_kwargs=config['env'],
            CollectorCls=Collector,
            TrajInfoCls=Traj,
            eval_env_kwargs=config['eval_env'],
            **config['sampler'])
    
    # OptimCls
    if optim == 'adam':
        Optim = torch.optim.Adam
    else:
        Optim = torch.optim.RMSprop

    algo = Algo(OptimCls=Optim, optim_kwargs=config['optim'], **config['algo'])
    agent = Agent(model_kwargs=config['model'], **config['agent'])
    return Runner(algo=algo, agent=agent, sampler=sampler, affinity=affinity, **config['runner'])


def _rl_train(out_dir, config_):
    config = ConfigFactory.from_dict(config_)
    async_ = config.get('affinity.async_sample')
    model_ = config.get('_k12.model.name')
    reset_ = config.get('_k12.sampler.mid_batch_reset')
    optim_ = config.get('_k12.optim.type')

    task = config.get('_k12.task')
    if task not in ('atari'):
        raise NotImplementedError(f'task: {task}')

    model_netw = config.get('_k12.model.network')
    if model_netw not in ('dqn'):
        raise NotImplementedError(f'network type: {model_netw}')

    sampl_mode = config.get('_k12.sampler.mode')
    if sampl_mode not in ('serial', 'cpu', 'gpu', 'alternating'):
        raise NotImplementedError(f'sampler mode: {sampl_mode}')

    if optim_ not in ('adam', 'rmsprop'):
        raise NotImplementedError(f'optimize type: {optim_}')

    runner = _rl_runner(task, async_, sampl_mode, model_netw, model_, optim_, reset_, config)

    with logger_context(out_dir, 'rl', 'k12', config):
        runner.train()


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

    try:
        with open(os.path.join(args.config_file), 'r') as f:
            config = json.load(f)

        if args.phase == 'train':
            _rl_train(args.out_dir, config)
        else:
            raise NotImplementedError(f'phase: {args.phase}')
    except Exception as err:
        logger.error('{}'.format(err))
        _err_msg(exc=True)
