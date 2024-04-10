from asyncio import transports
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import onpolicy
import torch.nn as nn
from .util import init
import copy
from .distributions import Categorical

def init_(m):
    init_method = nn.init.orthogonal_
    gain = nn.init.calculate_gain('relu')
    return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

def MLP(channels, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def Conv(use_orthogonal, activation_id):
    """ Multi-layer perceptron """
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
    gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])
    def init_(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
    out_size = int(64 / 8. * 64 / 8.)
    cnn_layers = [
        init_(nn.Conv2d(1, 8, 6, stride=2, padding=2)),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        init_(nn.Conv2d(8, 64, 6, stride=2, padding=2)),
        # nn.BatchNorm2d(128),
        nn.ReLU(),
        init_(nn.Conv2d(64, 64, 5, stride=1, padding=2)),
        # nn.BatchNorm2d(128),
        nn.ReLU(),
        init_(nn.Conv2d(64, 32, 6, stride=2, padding=2)),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        init_(nn.Conv2d(32, 16, 5, stride=1, padding=2)),
        # nn.BatchNorm2d(16),
        nn.ReLU(),
        Flatten(),
        init_(nn.Linear(out_size * 16, 256)),
        nn.LayerNorm(256),
        nn.ReLU(),
        init_(nn.Linear(256, 128)),
        nn.ReLU(),
        nn.LayerNorm(128),
        init_(nn.Linear(128, 64))]

    return nn.Sequential(*cnn_layers)

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class MLPAttention(nn.Module):
    def __init__(self, desc_dim, node_layer=False, matching_type=None):
        super().__init__()
        if matching_type == 'concat':
            self.mlp = MLP([desc_dim * 4, desc_dim, 1])
        elif node_layer:
            self.mlp = MLP([desc_dim * 3, desc_dim, desc_dim])
        else:
            self.mlp = MLP([desc_dim * 3, desc_dim, 1])
        self.node_layer = node_layer
        self.desc_dim = desc_dim
        self.matching_type = matching_type

    def forward(self, query, key, value, dist, mask, extra_dist=None):
        '''query: 1 x 128 x n_agent
        key: 1 x 128 x n_frontier
        dist: 1 x 128 x (n_agent x n_frontier)

        cat: 1 x 384 x (n_agent x n_frontier)

        value: 1 x 128 x n_frontier

        scores: 1 x n_agent x n_frontier

        output: n_agent x 128'''
        
        nq, nk = query.size(-1), key.size(-1)
        if self.matching_type == 'concat' and extra_dist is not None:
            scores = self.mlp(torch.cat((
                query.view(1, -1, nq, 1).repeat(1, 1, 1, nk).view(1, -1, nq * nk),
                key.view(1, -1, 1, nk).repeat(1, 1, nq, 1).view(1, -1, nq * nk),
                extra_dist, dist), dim=1)).view(1, nq, nk)
        elif self.node_layer:
            scores = self.mlp(torch.cat((
                query.view(1, -1, nq, 1).repeat(1, 1, 1, nk).view(1, -1, nq * nk),
                key.view(1, -1, 1, nk).repeat(1, 1, nq, 1).view(1, -1, nq * nk),
                dist), dim=1)).view(self.desc_dim, nq, nk)
        else:
            scores = self.mlp(torch.cat((
                query.view(1, -1, nq, 1).repeat(1, 1, 1, nk).view(1, -1, nq * nk),
                key.view(1, -1, 1, nk).repeat(1, 1, nq, 1).view(1, -1, nq * nk),
                dist), dim=1)).view(1, nq, nk)

        if mask is not None:
            if type(mask) is float:
                scores_detach = scores.detach()
                scale = torch.clamp(mask / (scores_detach.max(2).values - scores_detach.median(2).values), 1., 1e3)
                scores = scores * scale.unsqueeze(-1).repeat(1, 1, nk)
            else:
                scores = scores + (scores.min().detach() - 20) * (~mask).float().view(1, nq, nk)
        
        if self.node_layer:
            return None, scores
        if self.matching_type == 'multi':
            scores = scores * extra_dist.view(1, nq, nk)
        prob = scores.softmax(dim=-1)
        return torch.einsum('bnm,bdm->bdn', prob, value), scores
        
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def attention(self, query, key, value, mask):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        if mask is not None:
            scores = scores + (scores.min().detach() - 20) * (~mask).float().unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, 1, 1)
        prob = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), scores

    def forward(self, query, key, value, dist, mask):
        batch = query.shape[0]
        query, key, value = [l(x).view(batch, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, scores = self.attention(query, key, value, mask)
        return self.merge(x.contiguous().view(batch, self.dim*self.num_heads, -1)), scores.mean(1)

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, type: str, matching_type=None, node_layer=False):
        super().__init__()
        self.attn = MLPAttention(feature_dim, node_layer, matching_type) if type == 'cross' else MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.node_layer = node_layer

    def forward(self, x, source, dist, mask, extra_dis=None):
        if extra_dis is not None:
            message, weights = self.attn(x, source, source, dist, mask, extra_dis)
        else:
            message, weights = self.attn(x, source, source, dist, mask)
        if self.node_layer:
            return None, weights
        return self.mlp(torch.cat([x, message], dim=1)), weights

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, matching_type=None, use_double_matching=False, node_layer=False):
        super().__init__()
        self.phattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
        self.ghattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
        if node_layer:
            attn_list = [AttentionalPropagation(feature_dim, 4, type) for type in layer_names]
            attn_list[-1] = AttentionalPropagation(feature_dim, 4, layer_names[-1], node_layer=True)
            self.attn = nn.ModuleList(attn_list)
        elif use_double_matching:
            self.attn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, type, matching_type=matching_type) for type in layer_names])
        else:
            self.attn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, type) for type in layer_names])
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.names = layer_names
        self.use_double_matching = use_double_matching
        self.matching_type = matching_type
        self.node_layer = node_layer
        self.node_trans = MLP([feature_dim, feature_dim, 1])
        #self.score_layer = MLP([2*feature_dim, feature_dim, 1])

    def forward(self, desc0, desc1, desc2, desc3, dist, invalid=None, transport_matrix=None):
        # desc0: frontier
        # desc1: agent
        # desc2: agent_history
        # desc3: goal_history
        # fidx: n_frontier x 2
        if self.use_double_matching:
            if self.matching_type == 'multi':
                transport_matrix = self.node_trans(transport_matrix.transpose(1,2)).transpose(1,2)
            trans0 = transport_matrix.reshape(1, -1, desc1.size(-1) * desc0.size(-1))
            trans1 = transport_matrix.transpose(1, 2).reshape(1, -1, desc1.size(-1) * desc0.size(-1))
        else:
            trans0 = None
            trans1 = None
        dist0 = dist.reshape(1, -1, desc1.size(-1) * desc0.size(-1))
        dist1 = dist.transpose(1, 2).reshape(1, -1, desc1.size(-1) * desc0.size(-1))
        
        for idx, attn, phattn, ghattn, name in zip(range(len(self.names)), self.attn, self.phattn, self.ghattn, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            
            if name == 'cross' and self.use_double_matching:
                delta0, score0 = attn(desc0, src0, dist0, None, trans0)
                delta1, score1 = attn(desc1, src1, dist1, None, trans1)
            elif idx == len(self.names) - 1 and self.node_layer:
                delta0, score0 = attn(desc0, src0, dist0, None)
                delta1, score1 = attn(desc1, src1, dist1, None)
                return score1
            else:
                delta0, score0 = attn(desc0, src0, dist0, None)
                delta1, score1 = attn(desc1, src1, dist1, None)

            if name == 'cross':
                if desc2 is not None:
                    delta21, _ = phattn(desc2, desc1, None, None)
                    delta12, _ = phattn(desc1, desc2, None, None)
                    desc2 = desc2 + delta21
                else:
                    delta12 = 0
                if desc3 is not None:
                    delta03, _ = ghattn(desc0, desc3, None, None)
                    delta30, _ = ghattn(desc3, desc0, None, None)
                    desc3 = desc3 + delta30
                else:
                    delta03 = 0
                desc0, desc1 = (desc0 + delta0 + delta03), (desc1 + delta1 + delta12)
            else: 
                if desc2 is not None:
                    delta2, _ = phattn(desc2, desc2, None, None)
                    desc2 = desc2 + delta2
                if desc3 is not None:
                    delta3, _ = ghattn(desc3, desc3, None, None)
                    desc3 = desc3 + delta3
                desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

        # weights1: n_agent x n_frontier
        # fidx: n_agent x n_frontier x 2
        # assert (~invalid).any(1).all()
        scores = score1
        scores = log_optimal_transport(scores.log_softmax(dim=-2), self.bin_score, iters=5)[:, :-1, :-1]
        if invalid is not None:
            score_min = scores.min() - scores.max()
            scores = scores + (score_min - 40) * invalid.float()

        return scores * 15

class Perception_Graph(torch.nn.Module):
    def __init__(self, args, graph_linear):
        super(Perception_Graph, self).__init__()       
        self.num_agents = args.num_agents
        self.use_frontier_nodes = args.use_frontier_nodes
        self.use_double_matching = args.use_double_matching
        self.matching_type = args.matching_type
        feature_dim = 32
        layers = [32, 64, 128, 256]
        gnn_layers = 3 * ['self', 'cross']
        node_gnn_layers = ['self', 'cross']
        self.node_init = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.node_init[-1].bias, 0.0)
        self.dis_init = MLP([1, feature_dim, feature_dim])      
        self.gnn = AttentionalGNN(feature_dim, gnn_layers, use_double_matching=self.use_double_matching,matching_type=self.matching_type)
        if self.use_double_matching:
            self.node_gnn = AttentionalGNN(feature_dim, node_gnn_layers, node_layer=True)
        self.use_local_frontier = args.use_local_frontier
        self.cuda = args.cuda
        self.use_double_matching = args.use_double_matching
        
    def forward(self, observations, masks, frontier_graph_data_origin, agent_graph_data_origin): 
        frontier_graph_data = copy.deepcopy(frontier_graph_data_origin)
        last_frontier_data = copy.deepcopy(frontier_graph_data_origin)
        agent_graph_data = copy.deepcopy(agent_graph_data_origin)
        last_agent_data = copy.deepcopy(agent_graph_data_origin)
        if self.use_double_matching:
            node_graph_data = []
            last_node_data = []
            agent_pos_for_nodes = []
            last_agent_pos_for_nodes = []
        graph_agent_dis = []
        node_agent_dis = []
        global_step = int(observations['graph_last_pos_mask'][0].sum())
    
        for i in range(len(observations['graph_ghost_node_position'])):
            dis = []
            for a in range(self.num_agents):
                if self.use_frontier_nodes:
                    origin_dis = observations['graph_agent_dis'][i][a, :int(torch.sum(observations['graph_merge_frontier_mask'][i]))] 
                else:
                    origin_dis = observations['graph_agent_dis'][i][a][observations['graph_merge_ghost_mask'][i].reshape(-1)!=0]
                dis.append(self.dis_init(origin_dis.transpose(1,0).unsqueeze(0)).transpose(2,1))
            
            graph_agent_dis.append(torch.cat(dis,dim=0))
            if self.use_double_matching:
                graph_node_dis = []
                counts = int(torch.sum(observations['graph_node_pos'][i,:,-1]))
                for a in range(self.num_agents):
                    origin_dis = observations['agent_graph_node_dis'][i][a,:counts]
                    graph_node_dis.append(self.dis_init(origin_dis.transpose(1,0).unsqueeze(0)).transpose(2,1))
                node_position = observations['graph_node_pos'][i][:counts]
                node_position = self.node_init(node_position.transpose(0,1).unsqueeze(0))
                node_agent_dis.append(torch.cat(graph_node_dis,dim=0))
                last_node_position = observations['graph_last_node_position'][i, :global_step]
                last_node_position = self.node_init(last_node_position.transpose(0,1).unsqueeze(0))
                last_node_data.append(last_node_position)
                node_graph_data.append(node_position)
            
            if self.use_frontier_nodes:
                ghost_node_position = observations['graph_ghost_node_position'][i][:int(torch.sum(observations['graph_merge_frontier_mask'][i]))]
            else:
                ghost_node_position = observations['graph_ghost_node_position'][i][observations['graph_merge_ghost_mask'][i]!=0]
            agent_node_position = observations['agent_world_pos'][i]
            last_ghost_position = observations['graph_last_ghost_node_position'][i, :global_step]
            last_agent_position = observations['graph_last_agent_world_pos'][i, :global_step]
            ghost_node_position = self.node_init(ghost_node_position.transpose(0,1).unsqueeze(0))
            agent_node_position = self.node_init(agent_node_position.transpose(0,1).unsqueeze(0))
            last_ghost_position = self.node_init(last_ghost_position.transpose(0,1).unsqueeze(0)) 
            last_agent_position = self.node_init(last_agent_position.transpose(0,1).unsqueeze(0))
            if self.use_double_matching:
                agent_position_for_node = self.node_init(observations['agent_world_pos'][i].transpose(0,1).unsqueeze(0))
                last_agent_for_node = self.node_init(observations['graph_last_agent_world_pos'][i, :global_step].transpose(0,1).unsqueeze(0))
                agent_pos_for_nodes.append(agent_position_for_node)
                last_agent_pos_for_nodes.append(last_agent_for_node)
            
            frontier_graph_data[i].x = ghost_node_position
            agent_graph_data[i].x = agent_node_position
            last_frontier_data[i].x = last_ghost_position
            last_agent_data[i].x = last_agent_position
            
        if self.use_double_matching:
            all_node_edge = []
            for i in range(len(agent_graph_data)):
                e = self.node_gnn(node_graph_data[i], agent_pos_for_nodes[i], last_agent_pos_for_nodes[i] , last_node_data[i], node_agent_dis[i])
                all_node_edge.append(e)      
        
        all_edge = []
        for i in range(len(agent_graph_data)):
            if self.use_local_frontier:
                node_count = frontier_graph_data[i].x.shape[-1]
                pos = observations['graph_ghost_node_position'][i,:node_count,:2]*480
                agent_pos = observations['agent_world_pos'][i,:,:2]*480
                invalid = torch.zeros((self.num_agents, node_count))
                if self.cuda:
                    invalid = invalid.cuda()
                for a in range(self.num_agents):
                    dis = ((pos[:,0] - agent_pos[a, 0])**2 + (pos[:,1] - agent_pos[a, 1])**2)**0.5
                    invalid[a] = dis > 120
            else:
                invalid = None
            
            if self.use_double_matching:
                node_counts = int(torch.sum(observations['graph_node_pos'][i,:,-1]))
                ghost_counts = int(torch.sum(observations['graph_merge_ghost_mask'][i]!=0))
                transport_matrix = torch.zeros((self.num_agents, ghost_counts,32), dtype=torch.float).to(all_node_edge[i].device)
                temp_matrix = torch.zeros((node_counts, ghost_counts), dtype=torch.float).to(all_node_edge[i].device)
                idx = 0
                for node_idx in range(node_counts):
                    ghost_for_each_node = int(torch.sum(observations['graph_merge_ghost_mask'][i][node_idx]!=0))
                    temp_matrix[node_idx,idx:ghost_for_each_node+idx] = 1
                    idx += ghost_for_each_node
                for agent_id in range(self.num_agents):
                    temp_m = torch.einsum('ik, kj -> kji',all_node_edge[i][:,agent_id], temp_matrix)
                    transport_matrix[agent_id] = torch.sum(temp_m, dim=0)
            else:
                transport_matrix = None
            e = self.gnn(frontier_graph_data[i].x, agent_graph_data[i].x, last_agent_data[i].x , last_frontier_data[i].x, graph_agent_dis[i], invalid, transport_matrix)
            all_edge.append(e[0,0])

        return all_edge


class LinearAssignment(nn.Module):
    def __init__(self, args, device):
        super(LinearAssignment, self).__init__()
        self.num_agents = args.num_agents
        self.use_several_nodes = args.use_several_nodes
        self.device = device
    
    def forward(self, x, available_actions=None, deterministic=False):
        actions = []
        action_log_probs = []
        for i in range(len(x)):
            action_out = Categorical(x[i].shape[-1], x[i].shape[-1])
            action_logits = action_out(x[i].unsqueeze(0), available_actions, trans= False)
            if self.use_several_nodes:
                action = []
                j = 0
                while j<2:
                    action_tmp = action_logits.mode() if deterministic else action_logits.sample()
                    if action_tmp not in action:
                        action.append(action_tmp) 
                        j += 1
               
                action = torch.cat(action, dim=-1)
            else:
                action = action_logits.mode() if deterministic else action_logits.sample()
            action_log_prob = action_logits.log_probs(action)
            actions.append(action)
            action_log_probs.append(action_log_prob)
        
        return torch.cat(actions,0), torch.cat(action_log_probs,0)
    

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_log_probs = []
        dist_entropy = []
        for i in range(len(x)):
            action_out = Categorical(x[i].shape[-1], x[i].shape[-1])
            action_logits = action_out(x[i].unsqueeze(0), available_actions, trans= False)
            if self.use_several_nodes:
                action_log_tmp = []
                for j in range(2):
                    action_log_tmp.append(action_logits.log_probs(action[i,j:j+1].unsqueeze(0)))
                action_log_probs.append(torch.cat(action_log_tmp,dim=-1))
            else:
                action_log_probs.append(action_logits.log_probs(action[i].unsqueeze(0)))
            dist_entropy.append(action_logits.entropy().mean())

        return torch.cat(action_log_probs, 0),  torch.stack(dist_entropy, 0).mean()
    
    def optimal_transport(self, P, eps=1e-2):
        u = torch.zeros(P.shape[1], device=self.device)
        while torch.max(torch.abs(u-P.sum(0))) > eps:
            u = P.sum(0)
            P = P/(u.unsqueeze(0))
            P = P/(P.sum(1).unsqueeze(1))
        return P
