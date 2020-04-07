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
from k12ai.utils.log_parser import k12ai_set_phase
from k12ai.utils import k12ai_kill
from k12ai.common.log_message import MessageReport

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
from rlpyt.samplers.collections import TrajInfo
from rlpyt.samplers.async_.collectors import (DbCpuResetCollector, DbCpuWaitResetCollector)
from rlpyt.samplers.async_.collectors import (DbGpuResetCollector, DbGpuWaitResetCollector)
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector, CpuWaitResetCollector)
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)

# runner
from rlpyt.runners.async_rl import (AsyncRlEval, AsyncRl)
from rlpyt.runners.minibatch_rl import (MinibatchRlEval, MinibatchRl)
from k12ai.runners.RLEvaluate import (MinibatchRlEvalOnce, AsyncRlEvalOnce)

# algo & agent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.algos.dqn.r2d1 import R2D1

from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.qpg.ddpg import DDPG # noqa
from rlpyt.algos.qpg.sac import SAC # noqa
from rlpyt.algos.qpg.td3 import TD3 # noqa

from rlpyt.agents.qpg.ddpg_agent import DdpgAgent # noqa
from rlpyt.agents.qpg.sac_agent import SacAgent # noqa
from rlpyt.agents.qpg.td3_agent import Td3Agent # noqa

# atari
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1Agent
from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1AlternatingAgent
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.agents.pg.atari import AtariLstmAgent

# mujoco
from rlpyt.envs.gym import make as gym_make
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.agents.pg.mujoco import MujocoLstmAgent

_run_ID = '0'


def _signal_handler(sig, frame):
    if sig == signal.SIGUSR1:
        print("#######22##3333##################")
        MessageReport.status(MessageReport.ERROR, {'err_type': 'signal', 'err_text': 'handle quit signal'})


def _rl_check(config):
    async_ = config.get('affinity.async_sample', default=False)

    task = config.get('_k12.task')
    if task not in ('atari', 'mujoco'):
        raise NotImplementedError(f'task: {task}')

    netw = config.get('_k12.model.network')
    if netw not in ('dqn', 'pg'):
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
    elif task == 'mujoco':
        Env = gym_make
        Traj = TrajInfo
        config.put('agent', {})

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
    elif netw == 'pg':
        type_ = config.get('_k12.model.algo', default='none')
        if model == 'a2c':
            Algo = A2C
            if type_ == 'ff':
                Agent = AtariFfAgent if task == 'atari' else MujocoFfAgent
            elif type_ == 'lstm':
                Agent = AtariLstmAgent if task == 'atari' else MujocoLstmAgent
            else:
                raise NotImplementedError(f'algo type:{type_}')
        elif model == 'ppo':
            Algo = PPO
            if type_ == 'ff':
                Agent = AtariFfAgent if task == 'atari' else MujocoFfAgent
            elif type_ == 'lstm':
                Agent = AtariLstmAgent if task == 'atari' else MujocoLstmAgent
            else:
                raise NotImplementedError(f'algo type:{type_}')
        else:
            raise NotImplementedError(f'algo:{model}')
    elif netw == 'qpg':
        raise NotImplementedError(f'network:{netw}')

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

    if phase == 'train':
        algo = Algo(OptimCls=Optim, optim_kwargs=config['optim'], **config['algo'])
        agent = Agent(model_kwargs=config['model'], **config['agent'])
        if async_:
            Runner = AsyncRlEval if eval_ else AsyncRl
        else:
            Runner = MinibatchRlEval if eval_ else MinibatchRl
    elif phase == 'evaluate':
        snapshot_pth = os.path.join(context.LOG_DIR, f'run_{_run_ID}', 'params.pkl')
        if not os.path.exists(snapshot_pth):
            raise FileNotFoundError(f'model file: {snapshot_pth}')
        snapshot = torch.load(snapshot_pth)
        agent_state_dict = snapshot['agent_state_dict']
        if 'model' in agent_state_dict:
            agent_state_dict = agent_state_dict['model']
        optimizer_state_dict = snapshot['optimizer_state_dict']
        algo = Algo(OptimCls=Optim, optim_kwargs=config['optim'], **config['algo'],
                initial_optim_state_dict=optimizer_state_dict)
        agent = Agent(model_kwargs=config['model'], **config['agent'],
                initial_model_state_dict=agent_state_dict)
        Runner = AsyncRlEvalOnce if async_ else MinibatchRlEvalOnce

    return Runner(algo=algo, agent=agent, sampler=sampler, affinity=affinity, **config['runner'])


def _rl_train(config, phase):

    context.LOG_DIR = config.get('_k12.out_dir', default='/cache/output')

    runner = _rl_runner(config, phase)

    snapshot_mode = 'last'
    if phase == 'evaluate':
        snapshot_mode = 'none'

    with context.logger_context(context.LOG_DIR, _run_ID, 'k12', config, snapshot_mode):
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
            default='/cache/config.json',
            type=str,
            dest='config_file',
            help="config file")
    args = parser.parse_args()

    print('pid: {}'.format(os.getpid()))
    k12ai_set_phase(args.phase)
    MessageReport.status(MessageReport.RUNNING)
    try:
        if args.phase == 'train' or args.phase == 'evaluate':
            with open(os.path.join(args.config_file), 'r') as f:
                config = ConfigFactory.from_dict(json.load(f))
            _rl_train(config, args.phase)
        else:
            raise NotImplementedError(f'phase: {args.phase}')
        MessageReport.status(MessageReport.FINISH)
    except Exception:
        MessageReport.status(MessageReport.EXCEPT)
    finally:
        k12ai_kill(os.getpid())
