#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file main.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-01-19 17:34

import argparse
import os, signal
import json
import torch

from pyhocon import ConfigFactory

# K12AI
from k12rl.utils.log_parser import hzcsk12_log_message as _k12log

# utils
from rlpyt.utils.logging import context
from rlpyt.utils.launching.affinity import make_affinity

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
from rlpyt.runners.async_rl import (AsyncRlEval, AsyncRl)
from rlpyt.runners.minibatch_rl import (MinibatchRlEval, MinibatchRl)
from k12rl.runners.RLEvaluate import (MinibatchRlEvalOnce, AsyncRlEvalOnce)

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


def _signal_handler(sig, frame):
    if sig == signal.SIGUSR1:
        _k12log('k12rl_signal')


def _rl_check(config):
    async_ = config.get('affinity.async_sample', default=False)

    task = config.get('_k12.task')
    if task not in ('atari'):
        raise NotImplementedError(f'task: {task}')

    netw = config.get('_k12.model.network')
    if netw not in ('dqn'):
        raise NotImplementedError(f'network type: {netw}')

    mode = config.get('_k12.sampler.mode')
    if mode not in ('serial', 'cpu', 'gpu', 'alternating'):
        raise NotImplementedError(f'sampler mode: {mode}')

    optim = config.get('_k12.optim.type')
    if optim not in ('adam', 'rmsprop'):
        raise NotImplementedError(f'optimize type: {optim}')

    # TODO work around
    if async_ and mode != 'gpu' and not config.get('affinity.n_gpu', default=None):
        config.put('affinity.n_gpu', torch.cuda.device_count())

    return async_, task, netw, mode, optim


def _rl_runner(config, phase):
    async_, task, netw, mode, optim = _rl_check(config)
    model = config.get('_k12.model.name')
    reset = config.get('_k12.sampler.mid_batch_reset')
    alter = config.get('affinity.alternating', default=False)
    eval_ = config.get('_k12.runner.eval')

    if task == 'atari':
        Env = AtariEnv
        Traj = AtariTrajInfo

    affinity = make_affinity(**config['affinity'])

    if mode == 'serial':
        if async_:
            Sampler = AsyncSerialSampler
            Collector = DbCpuResetCollector if reset else DbCpuWaitResetCollector
        else:
            Sampler = SerialSampler
            Collector = CpuResetCollector if reset else CpuWaitResetCollector
    elif mode == 'cpu':
        if async_:
            Sampler = AsyncCpuSampler
            Collector = DbCpuResetCollector if reset else DbCpuWaitResetCollector
        else:
            Sampler = CpuSampler
            Collector = CpuResetCollector if reset else CpuWaitResetCollector
    elif mode == 'gpu':
        if async_:
            Sampler = AsyncAlternatingSampler if alter else AsyncGpuSampler
            Collector = DbGpuResetCollector if reset else DbGpuWaitResetCollector
        else:
            Sampler = AlternatingSampler if alter else GpuSampler
            Collector = GpuResetCollector if reset else GpuWaitResetCollector

    if phase == 'train':
        if async_:
            Runner = AsyncRlEval if eval_ else AsyncRl
        else:
            Runner = MinibatchRlEval if eval_ else MinibatchRl 
    elif phase == 'evaluate':
        Runner = AsyncRlEvalOnce if async_ else MinibatchRlEvalOnce

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


def _rl_train(out_dir, config, phase):

    runner = _rl_runner(config, phase)

    with context.logger_context(out_dir, 'rl', 'k12', config):
        runner.train()


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, _signal_handler)
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

    context.LOG_DIR = args.out_dir

    _k12log(f'k12rl_running:{args.phase}')
    try:
        if args.phase == 'train' or args.phase == 'evaluate':
            with open(os.path.join(args.config_file), 'r') as f:
                config = ConfigFactory.from_dict(json.load(f))
            _rl_train(context.LOG_DIR, config, args.phase)
        else:
            raise NotImplementedError(f'phase: {args.phase}')
    except Exception:
        _k12log('k12rl_except')
    _k12log('k12rl_finish')
