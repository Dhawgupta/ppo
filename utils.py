import numpy as np
import torch
import time 
import pickle as pkl
from gym.spaces import Box
from gym.envs.robotics.fetch.reach import FetchReachEnv
from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv

def collect_batch(batch_size, agent, env, l, gamma,stats, render  = False, scale_action = False):
    # print("Scale Action  {}".format(scale_action))
    stats['update_eps'].append(len(stats['time_at_end_eps']))
    batch = []
        # start collecting batch
    while len(batch) < batch_size - 1:
        # reset the env
        obs = env.reset()
        done = False
        actions = []
        log_probs = []
        values = []
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
            if scale_action:
                print("Scaling Value : {}".format(agent.network.scale_action))
                new_action = np.tanh( action ) * agent.network.scale_action
            else:
                new_action = action
            print("ACtion  : {}".format(new_action))
            new_obs, reward, done, info = env.step(np.array(new_action)) # changed to list for ur5
            new_obs = new_obs
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

    np.random.shuffle(batch)
    batch = batch[:batch_size]
    return batch



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
        ob, r, d, info = self.env.step(action)
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