import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase, MLPLayer
from onpolicy.algorithms.utils.mix import MIXBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.mantm import LinearAssignment
from onpolicy.utils.util import get_shape_from_obs_space

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead 
        self._recurrent_N = args.recurrent_N 
        self._grid_goal = args.grid_goal
        self._grid_goal_simpler = args.grid_goal_simpler
        self._grid_size = args.grid_size
        self._use_mgnn = args.use_mgnn
        self._use_batch_train = args.use_batch_train
        self._use_double_matching = args.use_double_matching
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, action_space, cnn_layers_params=args.cnn_layers_params, cnn_last_linear = not (self._grid_goal or self._grid_goal_simpler), graph_linear= False)
        else:
            self._mixed_obs = False
            self.base = CNNBase(args, obs_shape) if len(obs_shape)==3 else MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
        
        self.input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            self.input_size += self.hidden_size
        
        if self._use_mgnn and (not self._use_batch_train):
            self.act = LinearAssignment(args, device=device)
        else:
            self.act = ACTLayer(action_space, self.input_size, self._use_orthogonal, self._gain, args = args, device = device)

        if self._use_policy_vhead:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))
            if self._use_popart:
                self.v_out = init_(PopArt(self.input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(self.input_size, 1))

        self.to(device)

    def forward(self, obs, frontier_graph_data, agent_graph_data, rnn_states, masks, available_actions=None, available_actions_first=None, available_actions_second=None, deterministic=False):        
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if available_actions_first is not None:
            available_actions_first = check(available_actions_first).to(**self.tpdv)
        if available_actions_second is not None:
            available_actions_second = check(available_actions_second).to(**self.tpdv)
        
        actor_features = self.base(obs, masks, frontier_graph_data, agent_graph_data)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._use_mgnn:
            actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        else:
            
            actions, action_log_probs = self.act(actor_features, available_actions, available_actions_first, available_actions_second, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, frontier_graph_data, agent_graph_data, rnn_states, action, masks, available_actions=None, available_actions_first=None, available_actions_second=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if available_actions_first is not None:
            available_actions_first = check(available_actions_first).to(**self.tpdv)
        if available_actions_second is not None:
            available_actions_second = check(available_actions_second).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(obs, masks, frontier_graph_data, agent_graph_data)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._use_mgnn:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, available_actions_first, available_actions_second, active_masks = active_masks if self._use_policy_active_masks else None)

        values = self.v_out(actor_features) if self._use_policy_vhead else None
       
        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):        
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs, masks)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        values = self.v_out(actor_features)

        return values

class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, action_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal  
        self._activation_id = args.activation_id     
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._num_agents = args.num_agents
        self._use_mgnn = args.use_mgnn
        self.use_map_critic = args.use_map_critic
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if 'Dict' in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, share_obs_shape, action_space, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = CNNBase(args, share_obs_shape) if len(share_obs_shape)==3 else MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)

        self.input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(share_obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            self.input_size += self.hidden_size

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.input_size, args.n_rollout_threads, device=device))
        else:
            self.v_out = init_(nn.Linear(self.input_size, 1))

        self.to(device)

    def forward(self, share_obs, frontier_graph_data, agent_graph_data, rnn_states, masks, rank = None):
        if self._mixed_obs:
            for key in share_obs.keys():        
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(share_obs, masks, frontier_graph_data, agent_graph_data)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            
        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)
        
        if (self._use_mgnn and not self.use_map_critic):
            critic_features = torch.stack(critic_features, dim=0)
            critic_features = critic_features.reshape(critic_features.shape[0], -1)

        if self._use_popart:
            values = self.v_out(critic_features, rank)
        else:
            values = self.v_out(critic_features)

        return values, rnn_states
