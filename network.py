import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import  MultivariateNormal
import numpy as np
import torch.nn.functional as f


# TODO: create your network here. Your network should inherit from nn.Module.
# It is recommended that your policy and value networks not share the same core network. This can be
# done easily within the same class or you can create separate classes.

class PPONetwork(nn.Module):
    def __init__(self, action_count, in_size, scale_action = None,  action_limits = None):
        """
        Feel free to modify as you like.

        The policy should be parameterized by a normal distribution (torch.distributions.normal.Normal).
        To be clear your policy network should output the mean and stddev which are then fed into the Normal which
        can then be sampled. Care should be given to how your network outputs the stddev and how it is initialized.
        Hint: stddev should not be negative, but there are numerous ways to handle that. Large values of stddev will
        be problematic for learning.

        :param action_space: Action space of the environment. Gym action space. May have more than one action.
        :param in_size: Size of the input
        """
        super(PPONetwork, self).__init__()
        self.action_count = action_count
        self.in_size = in_size
        # self.actor = nn.Sequential ( nn.Linear(self.in_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.action_count ) )  # this will produce th
        self.actor = nn.Sequential ( nn.Linear(self.in_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.action_count ) )  # this will produce th

        # self.actor = nn.Sequential( nn.Linear(self.in_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.action_count, bias = False),nn.Tanh()) # this will produce the means
        self.std_model = nn.Sequential( nn.Linear(1,  1)) # std init to 0.5
        # nn.init.xavier_uniform_(self.actor[0].weight)
        # nn.init.xavier_uniform_(self.actor[2].weight)
        # nn.init.xavier_uniform_(self.actor[4].weight)

        # for w in self.actor.parameters():
        #     nn.init.xavier_uniform_(w)
        # nn.init.xavier_uniform_(self.actor.parameters())
        nn.init.constant_(self.std_model[0].weight, 0)
        nn.init.constant_(self.std_model[0].bias,   np.log(0.5))
        # nn.init.constant_(self.std_model[2].weight, 0)
        # nn.init.constant_(self.std_model[2].bias,   np.log(0.5))
        # # nn.init.constancleart_(self.actor[4].bias , 0)

        # self.critic = nn.Sequential( nn.Linear(self.in_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1) ) # the value function
        self.critic = nn.Sequential( nn.Linear(self.in_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1) ) # the value function

        # self.critic = nn.Sequential( nn.Linear(self.in_size, 64), nn.ReLU(), nn.Linear(64, 1) ) # the value function
        self.action_limits = action_limits
        self.scale_action = scale_action
        self.bool_scale = True # do we need to scale  actions ?
        if action_limits is None:
            if scale_action is None:
                self.bool_scale  = False # we dont
            else:
                self.action_limits = [ -self.scale_action, self.scale_action ]

    def get_action_count(self):
        return self.action_count

    def forward(self, inputs):
        # do a forward pass
        # print("Forward")
        # print("Input Shape :", inputs, inputs.shape)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
        mean = self.actor(inputs)
        # scale the mea
        std = self.std_model(torch.tensor([1], dtype = torch.double))
        # print("Mean :", mean, mean.shape)
        # print("Std : ",std, std.shape)
        std = torch.exp(std)
        # print("Exp Sttd : ", std, std.shape)
        value = self.critic(inputs)
        # print("Value  : ", value, value.shape)

        return mean, std, value

    def get_action(self, inputs):
        '''
        Returns the action for the bosercation, with no grad

        Returns:
            [action] -- float value
            probability --  real value
            value -- real value

        '''
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
            # print("Get Action | inputs : ",  inputs, inputs.shape)
            mean,  std, value = self.forward(inputs)
            # dist = Normal(mean, torch.ey
            # dist = Normal(mean, std)
            dist = MultivariateNormal(mean, std*torch.eye(self.action_count, dtype=torch.double))
            action = dist.sample()
            # print("get_action Action :", action, action.shape)
            # action = f.tanh(action)
            # print("get_action tanh(Action) :", action, action.shape  )
            # action = nn.Tanh(action)
            # dist = Normal(mean, std)
            # action = dist.sample()
            # scale the action later

            return action.tolist(), dist.log_prob(action).item(), value.item()


    def get_value(self, inputs):
        '''
        Returns the  value for observation
        '''
        # print("\n\n\n\n\nIn get_value")
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
            # print("Inputs : ", inputs,  inputs.shape)
            value = self.critic(inputs)
            # print("Value : ", value, value.shape)
            return value.item()

    def gaussian_parameters(self, inputs):
        """
        Return the mean and standard deviation

        Arguments:
            inputs {[type]} -- [description]
        """
        # print("Gaussian Parameterse")
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
            # print("Inputs : ", inputs, inputs.shape)
            mean, std , _ = self.forward(inputs)

            return mean , std

    def return_std(self, std):
        '''
        returns the correctly scalaed std
        '''
        return std* torch.eye(self.action_count, dtype = torch.double)
        

class PPONetwork2(nn.Module):
    def __init__(self, action_count, in_size, scale_action = None,  action_limits = None):
        """
        Feel free to modify as you like.

        The policy should be parameterized by a normal distribution (torch.distributions.normal.Normal).
        To be clear your policy network should output the mean and stddev which are then fed into the Normal which
        can then be sampled. Care should be given to how your network outputs the stddev and how it is initialized.
        Hint: stddev should not be negative, but there are numerous ways to handle that. Large values of stddev will
        be problematic for learning.

        :param action_space: Action space of the environment. Gym action space. May have more than one action.
        :param in_size: Size of the input
        """
        super(PPONetwork2, self).__init__()
        self.action_count = action_count
        self.in_size = in_size
        # self.actor = nn.Sequential ( nn.Linear(self.in_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.action_count ) )  # this will produce th
        self.actor = nn.Sequential ( nn.Linear(self.in_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, self.action_count*2 ) )  # this will produce th
       
        self.critic = nn.Sequential( nn.Linear(self.in_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1) ) # the value function

        self.action_limits = action_limits
        self.scale_action = scale_action
        self.bool_scale = True # do we need to scale  actions ?
        if action_limits is None:
            if scale_action is None:
                self.bool_scale  = False # we dont
            else:
                self.action_limits = [ -self.scale_action, self.scale_action ]

    def get_action_count(self):
        return self.action_count

    def forward(self, inputs):
        

        # do a forward pass
        # print("Forward")
        # print("Input Shape :", inputs, inputs.shape)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
        if len(inputs.shape) ==   1: 
            inputs = inputs.view([1,-1])
        output = self.actor(inputs)
        mean = output[:, :len(output[0])//2]
        std = output[:,len(output[0])//2:]

        # split the output into meaen and standard deviaiotn
        # scale the mea
        # std = self.std_model(torch.tensor([1], dtype = torch.double))
        # print("Mean :", mean, mean.shape)
        # print("Std : ",std, std.shape)
        std = torch.exp(std)
        # print("Exp Sttd : ", std, std.shape)
        value = self.critic(inputs)
        # print("Value  : ", value, value.shape)

        return mean, std, value

    def get_action(self, inputs):
        '''
        Returns the action for the bosercation, with no grad

        Returns:
            [action] -- float value
            probability --  real value
            value -- real value

        '''
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
            # print("Get Action | inputs : ",  inputs, inputs.shape)
            mean,  std, value = self.forward(inputs)
            # dist = Normal(mean, torch.ey
            # dist = Normal(mean, std)
            # print("Mean : ",  mean,  mean.shape)
            # print("Len Mean : ", len(mean))
            # print("Std: ", std,  std.shape)
            identity = torch.eye(self.action_count, dtype= torch.double)
            identity.unsqueeze_(0)
            identity = identity.expand( len(mean), self.action_count, self.action_count)
            std.unsqueeze_(1)
            stds = std*identity
            dist = MultivariateNormal(mean, stds)
            action = dist.sample()
            # print(action)

            action = action.view(-1)
            return action.tolist(), dist.log_prob(action).item(), value.item()
    
    
    def anmol(self):
        pass

    def get_value(self, inputs):
        '''
        Returns the  value for observation
        '''
        # print("\n\n\n\n\nIn get_value")
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
            # print("Inputs : ", inputs,  inputs.shape)
            value = self.critic(inputs)
            # print("Value : ", value, value.shape)
            return value.item()

    def gaussian_parameters(self, inputs):
        """
        Return the mean and standard deviation

        Arguments:
            inputs {[type]} -- [description]
        """
        # print("Gaussian Parameterse")
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy( np.array(inputs).reshape([-1]))
            # print("Inputs : ", inputs, inputs.shape)
            mean, std , _ = self.forward(inputs)

            return mean , std

    def return_std(self, std):
        '''
        returns the correctly scalaed std
        '''
        identity = torch.eye(self.action_count, dtype= torch.double)
        identity.unsqueeze_(0)
        identity = identity.expand( len(std), self.action_count, self.action_count)
        std = torch.unsqueeze(std, 1)
        stds = std*identity
        return  stds