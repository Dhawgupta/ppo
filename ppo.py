#!/usr/bin/env python

import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from ppo_agent import PPO
import gym
from network import PPONetwork, PPONetwork2
from ppo_agent import PPO
import numpy as np
import pickle as pkl
import time
from gym.spaces import Box
from gym.envs.robotics.fetch.reach import FetchReachEnv
# imports for the UR5
from utils import robotic_env_fn
# not going  to teh tf_seeds fucntion from  senseact.utils
from utils import str2bool


# from baselines.common.vec_env import SubprocVecEnv

import numpy as np
#env = gym.make('Pendulum-v0')
#from baselines.ppo2 import ppo2 as ppo
from utils import  env_fn, collect_batch


stats = dict()

def main(cycle_time, idn, baud, port_str, batch_size, mini_batch_size, epoch_count, gamma, l, max_action, outdir,
         ep_time, updates, optimizer, lr,scale_action  , env_type,normalize_observation , vfc, grad_clip_value, value_clip,  file_returns, grad_clip_norm, use_ss):
    
    stats['cycle_time'] = cycle_time
    stats['eps_len'] = ep_time
    stats['batch'] = batch_size
    stats['mini_batch'] = mini_batch_size
    stats['epoch_count'] = epoch_count
    stats['ep_time'] = ep_time
    stats['total_returns'] = [] #epsiode number along with the mean return in that
    stats['update_eps'] = [] # store the episode number for which the update strats i.e. 0 update will take eps from 0 to [1]
    stats['time_at_end_eps'] = [] # stores the time at the end of each episode
    stats['time_at_end_update'] = []
    stats['start_time'] = time.time()
    stats['eps_len'] = [] # store the length of each eps
    stats['normalize_observation'] = normalize_observation
    # stats['total_steps'] = 0
    stats['optimizer'] = optimizer
    stats['lr'] = lr
    tag = f"{time.time()}"
    summaries_dir = f"./summaries/{tag}"
    returns_dir = "./returns/"
    networks_dir = "./networks/"
    if outdir:
        summaries_dir = os.path.join(outdir, f"summaries/{tag}")
        returns_dir = os.path.join(outdir, "returns/")
        networks_dir = os.path.join(outdir, "networks/")

    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(returns_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=summaries_dir)
    # CHANGEME Change the env object declaration
    # env = ReacherEnv(cycle_time, ep_time, dxl.get_driver(False), idn, baud)
    # env = gym.make('FetchReach-v1')
    if env_type == 'reacher':
        env = gym.make('Reacher-v2')
    if env_type == 'fetch':
        env = env_fn()
    if env_type == 'pendulum':
        env = gym.make('Pendulum-v0')
    if env_type == 'mountaincar':
        env = gym.make('MountainCarContinuous-v0')
    if env_type == 'ur5':
        env = robotic_env_fn()
    
    # env = env_fn()
    # env = gym.make('Reacher-v2')
    # env = gym.make('Pendulum-v0')
    # obs_len = env.observation_space.shape[0]
    obs_len = env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CHANGEME : chnage the range  of action_limits
    actions_space = env.action_space.shape[0]
    print("Observation Length : {}\nAction Lenght : {}".format(obs_len, actions_space))
    nnet = None
    if not use_ss:
        print("Using state indepenednt standard deviation")
        nnet =  PPONetwork( action_count= actions_space, in_size= obs_len, action_limits = [-max_action,max_action], scale_action  = max_action)
    else:
        print("Using state dependent standard deviation")
        nnet =  PPONetwork2( action_count= actions_space, in_size= obs_len, action_limits = [-max_action,max_action], scale_action  = max_action)
    nnet.to(device)
    nnet = nnet.double()
    agent = PPO (device = device,  # cpu or cuda
                 network = nnet,  # your network
                 state_size = obs_len,  # size of your state vector
                 batch_size = batch_size,  # size of batch
                 mini_batch_size = mini_batch_size,
                 epoch_count = epoch_count,
                 summary_writer = summary_writer,
                 optimizer = optimizer,
                 lr = lr,
                 normalize_obs = normalize_observation,
                 vfc = vfc, 
                 grad_clip_value = grad_clip_value, 
                 value_clip = value_clip,
                 grad_clip_norm = grad_clip_norm,
                 )

    # This will  do this many updates

    for u in range(updates):
        batch = collect_batch(batch_size  = batch_size, agent  = agent, env = env, l =l , gamma = gamma,stats= stats, scale_action = scale_action  , file_returns = file_returns)

        # now we will run epochs on this batch
        # agent.learn_vectorized(batch)
        agent.learn_vectorized(batch)
        stats['time_at_end_update'].append(time.time() - agent.global_start_time)
        # save the agent state
        torch.save({
            'update': u,
            'model_state_dict': agent.network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            }, networks_dir + 'model_update_{}.pkl'.format(u))
        with open(returns_dir + "stats.pkl", 'wb') as f:
            pkl.dump(stats, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("Saved Network : {}".format(u))




    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_time", type=float, default=0.040, help="sense-act cycle time")
    parser.add_argument("--idn", type=int, default=1, help="Dynamixel ID")
    parser.add_argument("--baud", type=int, default=1000000, help="Dynamixel Baud")
    parser.add_argument("--port_str", type=str, default=None,
                        help="Default of None will use the first device it finds. Set this to override.")
    parser.add_argument("--batch_size", type=int, default=2000,
                        help="How many samples to record for each learning update")
    parser.add_argument("--mini_batch_size", type=int, default=40, help="Number of division to divide batch into")
    parser.add_argument("--epoch_count", type=int, default=20,
                        help="Number of times to train over the entire batch per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount")
    parser.add_argument("--l", type=float, default=0.95, help="lambda for lambda return")
    parser.add_argument("--max_action", type=float, default=0.3,
                        help="The maximum value you will output to the motor. "
                             "This should be dependent on the control mode which you select.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--ep_time", type=float, default=2.0, help="number of seconds to run for each episode.")
    parser.add_argument("--updates", type=int, default=400, help="Number of On Policy batch updates")
    parser.add_argument("--optimizer", type=str, default='adam', help="type of optimizer to use")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--scale_action",type = str2bool, default=False, help = "This will tell wether to scale the action or not")
    parser.add_argument('--env_type', type  = str, default  = 'reacher', help =  'env specification')
    parser.add_argument('--normalize_observation', type = str2bool, default  = False, help = 'Normalie the iobservation')
    parser.add_argument('--vfc',type = float, default = 1.0 , help = 'The value loss coeffecient')
    parser.add_argument('--grad_clip_value', type = float , default = None, help = 'The valuie by whiuch gradioent should be clipped None -> no clipping')
    parser.add_argument('--value_clip', type = float, default = None, help  = 'THer value by which we should clip the value loss')
    parser.add_argument('--file_returns',type = str, default = None,  help = "Enter a CSV  file name to store  returns")
    parser.add_argument('--grad_clip_norm' , type = float, default = None, help = "Clip the gradient norm")
    parser.add_argument('--use_ss', type = str2bool, default = True, help = "Bool to specify wether to use state dependent standard deviation")





    args = parser.parse_args()
    print(args)
    time.sleep(5)
    main(**args.__dict__)
    
