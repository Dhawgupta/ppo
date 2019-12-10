import numpy as np
import torch
import time 
import pickle as pkl
from gym.spaces import Box
from gym.envs.robotics.fetch.reach import FetchReachEnv
from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
import distutils
import csv
import os

def collect_batch(batch_size, agent, env, l, gamma,stats, render  = False, scale_action = False, file_returns = None):
    """This will be cused to collect a batch of the data
    
    Arguments:
        batch_size {int} -- [description]
        agent {PPO} -- [description]
        env {gym.env} -- [description]
        l {float} -- [description]
        gamma {float} -- [description]
        stats {dict} -- Store the statisitics ion the dictionary
    
    Keyword Arguments:
        render {bool} -- [description] (default: {False})
        scale_action {bool} -- scale the action for the enviornemnt (default: {False})
    
    Returns:
        list() -- return the batch
    """    
    # print("Scale Action  {}".format(scale_action))
    stats['update_eps'].append(len(stats['time_at_end_eps']))
    batch = []
    ep_returns = []  
        # start collecting batch
    while len(batch) < batch_size - 1:
        # reset the env
        obs = env.reset()
        done = False
        actions = []
        log_probs = []
        values = []
        obs = agent.normalize_observation(obs)
        observations = [obs]
        rewards = []
        stats['eps_len'].append(0)
        # print("OBs > {}".format(obs))
        while not done:
            stats['eps_len'][-1] +=1
            action, log_prob, value = agent.step(obs, done)
            # FIXME, had to use a list for action in Pendulum
            # testing the vectir code
            if render:
                try:
                    env.render()
                except:
                    pass
            actions.append(action)
            print(f"Scale {scale_action}")
            if scale_action:
                print("Scaling Value : {}".format(agent.network.scale_action))
                new_action = np.tanh( action ) * agent.network.scale_action
            else:
                print("Not scaling action")
                new_action = action
            print("ACtion  : {}".format(new_action))
            new_obs, reward, done, info = env.step(new_action) # changed to list for ur5
            new_obs = agent.normalize_observation( new_obs)
            observations.append(new_obs)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            # print("{:<30} | {:<30} | {:30} | {:<30} | {}".format(str(obs), str(new_action), str(reward), str(new_obs), done))
            obs = new_obs
        env.reset() # reset to stop the motor from turning unnceseeatly
        values.append(agent.network.get_value(obs))
        # Compute the returns
        agent.summary_writer.add_scalar("Episode/Return", np.sum(rewards), len(stats['total_returns']) )
        stats['total_returns'].append(np.sum(rewards))
        stats['time_at_end_eps'].append(time.time() -  agent.global_start_time)
        G,Gl, H = agent.compute_return(r_buffer =  rewards,v_buffer =  values, l  = l, gamma = gamma)
        # save the returns into the csv file
        ep_returns.append(np.mean(rewards))
        
        # Put them in the batchs
        #append transition to the batch
        for t in range(len(G)):
            batch.append(
                {
                    0: observations[t],
                    1: actions[t],
                    2: observations[t+1],
                    3: G[t],
                    4: Gl[t],
                    5: H[t],
                    6: log_probs[t],
                    7: values[t],

                }
            )
    if file_returns is not None:
        # if not os.path.exists('./results/' +file_returns):
        #     with open("./results/" + file_returns, 'w') as f:
        #         wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        #         wr.writerow(np.sum(rewards))
            
        with open("./results/" + file_returns, 'a') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(ep_returns)

    np.random.shuffle(batch)
    batch = batch[:batch_size]
    return batch


def normalize_observation(obs, rms) : 
    '''
    '''
    return rms.update(obs)





class RunningMeanSTD(object):
    def __init__(self, obs_size, clips = [-5.0, 5.0]):
        # this will be used to maintinta the running mean sand the standartdd deviation
        self.__mean = np.zeros(obs_size)
        self.__var = np.zeros(obs_size)
        self.__no_obs = 0
        self.__clips = clips
    
    def update(self,obs):
        '''
        Return the normalized observation 
        and update the parameters appropriately
        '''
        print("Normalizing Observation")
        lentype = len(obs.shape)
        obs = obs.reshape([-1])
        # update the statistics
        self.__mean  = self.__mean*(self.__no_obs / ( 1 + self.__no_obs)) + obs * (1 / (1 + self.__no_obs))
        self.__var = self.__var*(self.__no_obs / ( 1 + self.__no_obs)) + ( (self.__mean - obs)**2 )*(1 / (1 + self.__no_obs) )
        self.__no_obs +=1 
        # normalzie the observation
        new_obs = (obs - self.__mean)/ ( np.sqrt(self.__var) + 1e-5*np.ones_like(self.__mean))
        # clip the observation
        if lentype == 1:
            return np.clip(new_obs, np.min(self.__clips), np.max(self.__clips))
        else:
            return np.clip(new_obs.reshape([1,-1]), np.min(self.__clips), np.max(self.__clips))


        

class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.spec = env.spec
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(13,)) #gym.env.observation_space['observation']
        self.action_space = env.action_space
        self.ep_steps = 0
        self.ep_reward = 0
        self.max_steps = 100 if self.env.reward_type == 'dense' else 300

    def reset(self):
        self.ep_steps = 0
        self.ep_reward = 0
        ob = self.env.reset()
        return np.concatenate([ob['observation'], ob['desired_goal']])


    def step(self, action):
        '''
        action entered will probably be a scalar 
        convert it to a list
        '''
        
        ob, r, d, info = self.env.step(np.array(action)) # converted the aciton into an array
        self.ep_steps += 1
        self.ep_reward += r
        if self.env.reward_type == 'sparse':
            if r == 0:
                r = 1
                d = True
            elif r == -1:
                r = 0
        if d or self.ep_steps >= self.max_steps:
            info['episode'] = {}
            info['episode']['r'] = self.ep_reward
            info['episode']['l'] = self.ep_steps
            d = True
        return np.concatenate([ob['observation'], ob['desired_goal']]), r, d, info

    def close(self):
        self.env.close()

def env_fn():
    env = FetchReachEnv(reward_type='dense')
    return EnvWrapper(env)


class UR5EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.spec = env.spec
        self.observation_space = env.observation_space 
        self.action_space = env.action_space
        self.ep_steps = 0
        self.ep_reward = 0
        #self.max_steps = 100 if self.env.reward_type == 'dense' else 300

    def reset(self):
        self.ep_steps = 0
        self.ep_reward = 0
        return self.env.reset()


    def step(self, action):
        ob, r, d, info = self.env.step(action)
        self.ep_steps += 1
        self.ep_reward += r
        if d:
            info['episode'] = {}
            info['episode']['r'] = self.ep_reward
            info['episode']['l'] = self.ep_steps
        return ob, r, d, info

    def close(self):
        self.env.close()


def robotic_env():
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))
    env = ReacherEnv(
        setup="UR5_default",
        host='169.254.39.68',
        dof=2,
        control_type="velocity",
        target_type="position",
        reset_type="zero",
        reward_type="precision",
        derivative_type="none",
        deriv_action_max=5,
        first_deriv_max=2,
        accel_max=1.4,
        speed_max=0.3,
        speedj_a=1.4,
        episode_length_time=4.0,
        episode_length_step=None,
        actuation_sync_period=1,
        dt=0.04,
        #run_mode="multiprocess",
        run_mode='singlethread',
        rllab_box=False,
        movej_t=2.0,
        delay=0.0,
        random_state=rand_state
        )
    env = NormalizedEnv(env)
    env.start()
    return env

def robotic_env_fn():
    env = robotic_env()
    return UR5EnvWrapper(env)

def str2bool(v):
    return bool(distutils.util.strtobool(v))