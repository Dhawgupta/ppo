"""
Place your PPO agent code in here.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
import time
import sys
from torch.distributions.multivariate_normal import  MultivariateNormal


class PPO:
    def __init__(self,
                 device,  # cpu or cuda
                 network,  # your network
                 state_size,  # size of your state vector
                 batch_size,  # size of batch
                 mini_batch_size,
                 epoch_count,
                 gamma=0.99,  # discounting
                 l=0.95,  # lambda used in lambda-return
                 eps=0.2,  # epsilon value used in PPO clipping
                 summary_writer: SummaryWriter = None,
                 optimizer = 'adam',
                 lr = 0.001):
        self.device = device

        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.epoch_count = epoch_count
        self.gamma = gamma
        self.l = l
        self.eps = eps
        self.summary_writer = summary_writer
        self.global_start_time = time.time()
        self.state_size = state_size
        self.network = network
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 0.0001)
        print("Using optimizer {} with lr = {}".format(optimizer, lr))
        if optimizer.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr = lr)
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr)
        print(self.optimizer)
        self.counter = 0
        self.updates_to_policy = 0 # this will keep a track on  the number of updates on policy

    def step(self, state,  terminal):
        """
        You will need some step function which returns the action.
        This is where I saved my transition data in my own code.
        :param state: the observation
        :param r: the reward we got
        :param terminal: bool value
        :return:
        """
        # print("In Step : ")
        if not terminal:
            if not isinstance(state, torch.Tensor):
                state = np.array(state).reshape([-1])
                state = torch.tensor(state, dtype=torch.double)
            # print("State : ", state, state.shape)
            action, log_prob, value = self.network.get_action(state)
            # print("Action : ", action, action.shape)
            # print("Log Prob : ", log_prob, log_prob.shape)
            # print("Value : ", value, value.shape)
            return action, log_prob, value
        else:
            raise Exception("Terminal state reached")


    @staticmethod
    def compute_return(r_buffer, v_buffer, l, gamma):
        """

        Compute the return. Unit test this function

        :param r_buffer: rewards
        :param v_buffer: values
        :param l: lambda value
        :param gamma: gamma value
        :return:  the return
        """
        # print("Compute Return")
        with torch.no_grad():
            # print("r_buffer", r_buffer, len(r_buffer))
            # print("v_buffer", v_buffer, len(v_buffer))
            # print(" L and gamma" ,l,gamma)
            T = len(r_buffer)
            returns = np.zeros([T])
            lambda_returns = np.zeros([T])
            advantages = np.zeros([T])
            G = 0
            G_lambda = 0
            H = 0
            for t in range(T-1, -1, -1):
                G = r_buffer[t] + gamma*G
                G_lambda = r_buffer[t] + gamma*( (1 - l)*v_buffer[t+1] + l*G_lambda )
                H = G_lambda - v_buffer[t]
                returns[t] = G
                lambda_returns[t] = G_lambda
                advantages[t] = H
            # normalize the advanteage
            mu = np.mean(advantages)
            std = np.std(advantages)
            # print("Advantages", advantages)
            advantages = (advantages - mu)/( std + 1e-5)
            # print("Advantages", advantages)
        return returns, lambda_returns, advantages

    def compute_advantage(self, g, v):
        """
        Compute the advantage
        :return: the advantage
        """
        # print("G ", g, g.shape)
        return g - v

    def compute_rho(self, action, logold_pi, new_pi):
        """
        Compute the ratio between old and new pi
        :param actions: teh aciotn a saclaer
        :param old_pi: log probability of action
        :param new_pi: the new distribution
        :return:
        """
        action = torch.tensor([action], dtype = torch.double)
        # print("Computer Rho, action ", action, action.item())
        # print("New action log : ", new_pi.log_prob(action), new_pi.log_prob(action).shape)
        return torch.exp( new_pi.log_prob(action) - logold_pi)

    def compute_rho_vec(self, action, log_pi_old, new_pi):
        log_pi_old = log_pi_old.view([-1,1])
        # print("Log pi Old " ,log_pi_old, log_pi_old.shape)
        log_pi_new = new_pi.log_prob(action).view([-1,1])
        # print("Log Pi New  " , log_pi_new, log_pi_new.shape)
        ratio = log_pi_new - log_pi_old
        # print("Ratio  : " , ratio, ratio.shape)
        return torch.exp(ratio)


    def learn_vectorized(self, batch):
        """
        Here's where you should do your learning and logging.
        THe  batch is shuffled
        :param t: The total number of transitions observed.

        :return:
        """
        # print("Learn")
        for e in range(self.epoch_count):
            self.mini_batch_no = 0
            avg_clips = 0
            while True:
                try:
                    mini_batch = self.get_mini_batch(batch)
                    self.mini_batch_no += 1
                except Exception as exp:
                    print(exp)
                    break

                # get  a  vectro of different qunatitiress
                obs_ts = []
                obs_ts_1 = []
                lambda_returns = []
                advantages = []
                log_probs_old = []
                actions = []

                # actions = torch.empty()
                for t in mini_batch:
                    obs_ts.append(t[0])
                    obs_ts_1.append(t[2])
                    actions.append(t[1])
                    log_probs_old.append(t[6])
                    lambda_returns.append(t[4])
                    advantages.append(t[5])
               #  make a tensor out of everything
                obs_ts = torch.tensor(obs_ts, dtype = torch.double)
                obs_ts_1 = torch.tensor(obs_ts_1, dtype=torch.double)
                actions = torch.tensor(actions, dtype=torch.double).view(size = [-1, self.network.get_action_count() ])
                log_probs_old = torch.tensor(log_probs_old, dtype=torch.double).view(size  = [-1,1])
                lambda_returns = torch.tensor(lambda_returns, dtype=torch.double).view(size =  [-1,1])
                advantages = torch.tensor(advantages, dtype=torch.double).view(size  = [-1,1])
                # print("Details about different vectors")
                # print("OBsts : ", obs_ts.shape)
                # print("OBsts +1 : ", obs_ts_1.shape)
                # print("ACtions : ", actions.shape)
                # print("Log Probability old : ", log_probs_old.shape)
                # print("lmabda Returns : ", lambda_returns.shape)
                # print("advantages : ", advantages.shape)
                # print("Minibatch  ", mini_batch, len(mini_batch))
                # now we have the mini batch
                self.optimizer.zero_grad()
                # self.loss_combined = torch.tensor
                # loss_value_cumulative = torch.tensor([0], dtype = torch.double).detach()
                #     mean, std, value = self.network(torch.from_numpy(obs_t))

                mean, std, value = self.network(obs_ts)
                # print("Mean : ", mean,  mean.shape)
                # print("Std : ", std, std.shape)
                # print("Value :  ", value, value.shape)
                std = self.network.return_std(std)
                dist = MultivariateNormal(mean, std)
                # print("Dist : ", dist)
                action = dist.sample()
                # print("Action :", action, action.shape)
                log_prob = dist.log_prob(action)
                # print("Log prob", log_prob, log_prob.shape)
                mseloss = nn.MSELoss()
                rhop = self.compute_rho_vec(actions, log_probs_old, dist)

                # print("Rhop ", rhop, rhop.shape)
                self.summary_writer.add_scalar("STD",  torch.norm(std).item(), self.updates_to_policy)
                rhop_clipped = torch.clamp(rhop, min = 1  - self.eps, max = 1 + self.eps)
                rhopc = rhop_clipped.clone()
                clipped = np.sum(rhopc.detach().numpy() == 1+self.eps) + np.sum(rhopc.detach().numpy() == 1-self.eps)
                # print("Clipped : {}%".format((clipped/len(rhopc)) *100 )  )
                # print("Rho Cliiped : ", rhop_clipped, rhop_clipped.shape )
                avg_clips += clipped
                # print("Advantaese : ", advantages, advantages.shape)
                obj1 = advantages*rhop
                obj2 = advantages*rhop_clipped
                # print("Obj1 : ", obj1, obj1.shape)
                # print("Obj2 : ",  obj2, obj2.shape)
                minobj = torch.min(obj1, obj2)
                # print("Min Objective  :  ", minobj, minobj.shape)
                loss_policy = - minobj.mean()
                # loss_policy2 = - minobj.mean().view([1])
                # print("Lambda Returns  : ", lambda_returns, lambda_returns.shape)
                # print("Value  :", value, value.shape)

                loss_value = mseloss(lambda_returns, value)
                # print("Loss Value : ", loss_value, loss_value.shape)
                # print("Loss Policy : ", loss_policy , loss_policy.shape)
                self.summary_writer.add_scalar("Losses/Value", loss_value.item(), self.updates_to_policy)
                self.summary_writer.add_scalar("Losses/Policy", loss_policy.item(),self.updates_to_policy)
                loss_combined = 0.1 * loss_policy + loss_value
                # print("Loss combined : ", loss_combined, loss_combined.shape)
                loss_combined.backward()
                with torch.no_grad():

                    for name, weight in self.network.named_parameters():
                        self.summary_writer.add_scalar(f"Gradients/{name}", torch.norm(weight.grad).item() ,  self.updates_to_policy )

                self.updates_to_policy +=1
                self.optimizer.step()
            
            self.summary_writer.add_scalar("Clips/Clipped",(  avg_clips/self.batch_size)*100 ,self.updates_to_policy)

            print("Epoch : {}/{}".format(e, self.epoch_count))




    def get_mini_batch(self, batch):
        mini_batch = []
        if self.mini_batch_no * self.mini_batch_size < len(batch):
            for i in range(self.mini_batch_no * self.mini_batch_size , min(len(batch),(self.mini_batch_no +1) * self.mini_batch_size  )):
                mini_batch.append(batch[i])
        else:
            raise Exception("Batch Complete")
        return mini_batch
