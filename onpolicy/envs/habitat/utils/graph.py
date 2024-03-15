import math
import numpy as np
import torch
import onpolicy
from .predictor import Predictor
from . import pose as pu
from icecream import ic
from onpolicy.envs.habitat.model.PCL.resnet_pcl import resnet18
from torch import nn
from onpolicy.envs.habitat.utils.extractor import VisualEncoder

class Node(object):
    def __init__(self, info=None):
        self.node_num = None
        self.time_t = None
        self.neighbors = []
        self.neighbors_node_num = []
        self.embedding = None
        self.misc_info = None
        self.action = -1
        self.visited_time = []
        self.visited_memory = []
        if info is not None:
            for k, v in info.items():
                setattr(self, k, v)

class Graph(object):
    def __init__(self, memory_size, num_agents, input_shape, feature_dim):
        self.num_agents = num_agents
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.M = memory_size
    
    def num_node(self, b):
        return len(self.node_position_list[b])

    def num_node_max(self):
        return self.graph_mask.sum(axis=1).max().astype(np.compat.long)

    def reset(self, num_agents):
        if num_agents: self.num_agents = num_agents
        self.node_position_list = [[] for _ in range(self.num_agents)] # This position list is only for visualizations
        self.graph_memory = np.zeros([self.num_agents, self.M, self.feature_dim])
        self.graph_act_memory = np.zeros([self.num_agents, self.M],dtype=np.uint8)
        self.A = np.zeros([self.num_agents, self.M, self.M],dtype=np.bool_)
        self.graph_mask = np.zeros([self.num_agents, self.M])
        self.graph_time = np.zeros([self.num_agents, self.M])
        self.graph_id_trace = np.zeros([self.num_agents, self.M])
        self.last_localized_node_idx = np.zeros([self.num_agents], dtype=np.int32)
        self.last_local_node_num = np.zeros([self.num_agents])
        self.last_localized_node_embedding = np.zeros([self.num_agents, self.feature_dim], dtype=np.float32)

    def reset_at(self,b):
        self.graph_memory[b] = 0
        self.graph_act_memory[b] = 0
        self.A[b] = 0
        self.graph_mask[b] = 0
        self.graph_time[b] = 0
        self.last_localized_node_idx[b] = 0
        self.node_position_list[b] = []

    def initialize_graph(self, b, new_embeddings, positions):
        self.add_node(b, node_idx=0, embedding=new_embeddings[b], time_step=0, position=positions[b])
        self.record_localized_state(b, node_idx=0, embedding=new_embeddings[b])

    def add_node(self, b, node_idx, embedding, time_step, position):
        self.node_position_list[b].append(position)
        self.graph_memory[b, node_idx] = embedding
        self.graph_mask[b, node_idx] = 1.0
        self.graph_time[b, node_idx] = time_step

    def record_localized_state(self, b, node_idx, embedding):
        self.last_localized_node_idx[b] = node_idx
        self.last_localized_node_embedding[b] = embedding
        self.graph_id_trace[b][node_idx] = b+1

    def add_edge(self, b, node_idx_a, node_idx_b):
        self.A[b, node_idx_a, node_idx_b] = 1.0
        self.A[b, node_idx_b, node_idx_a] = 1.0
        return

    def update_node(self, b, node_idx, time_info, embedding=None):
        if embedding is not None:
            self.graph_memory[b, node_idx] = embedding
        self.graph_time[b, node_idx] = time_info
        return

    def update_nodes(self, bs, node_indices, time_infos, embeddings=None):
        if embeddings is not None:
            self.graph_memory[bs, node_indices] = embeddings

        self.graph_time[bs, node_indices.astype(np.long)] = time_infos

    def get_positions(self, b, a=None):
        if a is None:
            return self.node_position_list[b]
        else:
            return self.node_position_list[b][a]

    def get_neighbor(self, b, node_idx, return_mask=False):
        if return_mask: return self.A[b, node_idx]
        else: return np.where(self.A[b, node_idx])[0]

    def calculate_multihop(self, hop):
        return np.matrix_power(self.A[:, :self.num_node_max(), :self.num_node_max()].float(), hop)
    

class Merge_Graph(object):
    def __init__(self, args, memory_size, num_agents, input_shape, feature_dim):
        self.args = args
        self.memory = None
        self.memory_mask = None
        self.memory_time = None
        self.memory_num = 0
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.M = memory_size
        self.num_agents = num_agents
        self.map_size_cm = args.map_size_cm
        self.resolution = args.map_resolution
        self.ghost_node_size = args.ghost_node_size
        self.angles = [0, 30, 60, 90, 120, 150, 180, -35, -60, -90, -120, -150]
        self.processor = self.build_processor()
        
    def build_processor(self):
        visual_encoder = resnet18(num_classes=self.feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        ckpt_pth = onpolicy.__path__[0]+ "/envs/habitat/model/PCL/PCL_encoder.pth"
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval()
        return visual_encoder

    def num_node(self):
        return len(self.node_position_list)

    def num_node_max(self):
        return self.graph_mask.sum(axis=0).max().astype(np.long)

    def reset(self):
        self.node_position_list = [] # This position list is only for visualizations
        self.ghost_node_position = np.zeros([self.M, self.ghost_node_size, 3])
        self.ghost_node_feature = np.zeros([self.M, self.ghost_node_size, self.feature_dim])
        self.ghost_mask = np.zeros([self.M, self.ghost_node_size], dtype = np.int32)
        self.graph_memory = np.zeros([self.M, self.feature_dim])
        self.graph_act_memory = np.zeros([self.M],dtype=np.uint8)
        self.graph_creat_id = np.zeros([self.M],dtype=np.uint8)
        self.A = np.zeros([self.M, self.M],dtype=np.float64)
        self.D = np.zeros([self.M, self.M],dtype=np.float64)
        self.graph_mask = np.zeros([self.M])
        self.graph_time = np.zeros([self.M])
        self.last_localized_node_idx = np.zeros([self.num_agents], dtype=np.int32)
        self.last_local_node_num = np.zeros([1])
        self.last_localized_node_embedding = np.zeros([self.num_agents, self.feature_dim], dtype=np.float32)
        self.num_init_nodes = 0
        self.flag_reset = False

    def reset_at(self):
        self.graph_memory = np.zeros([self.M, self.feature_dim])
        self.graph_act_memory = np.zeros([self.M],dtype=np.uint8)
        self.A = np.zeros([self.M, self.M],dtype=float)
        self.D = np.zeros([self.M, self.M],dtype=float)
        self.graph_mask = np.zeros([self.M])
        self.graph_time = np.zeros([self.M])
        self.ghost_mask = np.zeros([self.M, self.ghost_node_size], dtype=np.int32)
        self.last_localized_node_idx = np.zeros([self.num_agents], dtype=np.int32)
        self.node_position_list = []
        self.ghost_node_position = np.zeros([self.M, self.ghost_node_size, 3])
        self.ghost_node_feature = np.zeros([self.M, self.ghost_node_size, self.feature_dim])
        self.graph_creat_id = np.zeros([self.M],dtype=np.uint8)
        self.num_init_nodes = 0
        self.flag_reset = False

    def initialize_graph(self, b, new_embeddings, positions):
        self.add_node(node_idx=b, agent_id=b, embedding=new_embeddings, time_step=0, position=positions)
        self.record_localized_state(node_idx=b, agent_id=b, embedding=new_embeddings)

    def add_node(self, node_idx, agent_id, embedding, time_step, position):
        self.node_position_list.append(position)
        self.graph_memory[node_idx] = embedding
        self.graph_mask[node_idx] = 1.0
        self.graph_time[node_idx] = time_step
        self.graph_creat_id[node_idx] = agent_id+1
    
    def extract_feature(self, img, relative_ori):
        if relative_ori < -180 or relative_ori > 180:
            r = relative_ori//180
            relative_ori += 180*(-r)
        block_size = 360/len(self.angles)
        anchor = img.shape[-1]/360*relative_ori+img.shape[-1]/2
        bound = math.ceil(anchor-block_size/360*img.shape[-1])
        size = math.ceil(2*block_size/360*img.shape[-1])
        if bound < 0: bound += img.shape[-1]
        if bound + size <= img.shape[-1]:
            part_img = img[:,:,bound:bound + size]
        else:
            part_img = torch.zeros([img.shape[0], img.shape[1], size])
            part_img[:,:,:img.shape[-1]-bound] = img[:,:,bound:]
            part_img[:,:,img.shape[-1]-bound:] = img[:,:,:size-(img.shape[-1]-bound)]
        part_img = part_img.unsqueeze(0)
        img_feature = nn.functional.normalize(self.processor(part_img)).detach() 
        return img_feature.squeeze().detach().cpu().numpy()

    def add_ghost_node(self, agent_id, labels, position, imgs, positions, raw_img=None):
        for i in range(len(labels)) :
            if labels[i] == 1:
                _, is_localized = self.if_nearby(self.node_position_list, position[i])
                angle = math.pi/len(self.angles)
                ghost_th = 1 * math.sin(angle) * 2 - 0.05
                self.ghost_node_position[self.last_localized_node_idx[agent_id],i] = position[i]
                if not is_localized: 
                    self.ghost_mask[self.last_localized_node_idx[agent_id], i] = 1

    def record_localized_state(self, node_idx, agent_id, embedding):
        self.last_localized_node_idx[agent_id] = node_idx
        self.last_localized_node_embedding[agent_id] = embedding

    def add_edge(self, node_idx_a, node_idx_b, dis= None):
        if self.args.use_edge_info is not None:    
            relative = int((self.node_position_list[node_idx_a][-1] - self.node_position_list[node_idx_b][-1] + 180.0) / 5.)
            invert_relative = int((self.node_position_list[node_idx_b][-1] - self.node_position_list[node_idx_a][-1] + 180.0) / 5.)
            self.A[node_idx_a, node_idx_b] = relative
            self.A[node_idx_b, node_idx_a] = invert_relative
        else:
            self.A[node_idx_a, node_idx_b] = 1.0
            self.A[node_idx_b, node_idx_a] = 1.0
        
        return

    def update_node(self, node_idx, agent_id, time_info, embedding=None):
        if embedding is not None:
            self.graph_memory[node_idx] = embedding
        self.graph_time[node_idx] = time_info
        return

    def update_nodes(self, node_indices, agent_id, time_infos, embeddings=None):
        if embeddings is not None:
            self.graph_memory[node_indices] = embeddings
        self.graph_time[node_indices] = time_infos

    def get_positions(self, b, a=None):
        if a is None:
            return self.node_position_list[b]
        else:
            return self.node_position_list[b][a]
    
    def get_ghost_positions(self, x, agent_id):
        valid_ghost_position = self.ghost_node_position[self.ghost_mask == 1]
        index_i, index_j = np.where(self.ghost_mask == 1)
        position = valid_ghost_position[x]
        return position
    
    def get_all_ghost_positions(self):
        return self.ghost_node_position[self.ghost_mask == 1]

    def get_neighbor(self, b, node_idx, return_mask=False):
        if return_mask: return self.A[b, node_idx]
        else: return np.where(self.A[b, node_idx])[0]

    def calculate_multihop(self, hop):
        return np.matrix_power(self.A[ :self.num_node_max(), :self.num_node_max()].float(), hop)
    
    def if_nearby(self, position_A_list, position_B, ghost= None, target_dis = 0.8):
        mini_idx = []
        is_localized = False
        if type(position_A_list) == np.ndarray:
            for i in range(position_A_list.shape[0]):
                for j in range(position_A_list.shape[1]):
                    if self.ghost_mask[i,j] == 0:
                        continue
                    else:
                        dis = pu.get_l2_distance(position_A_list[i,j][0], position_B[0], \
                                                position_A_list[i,j][1], position_B[1])
                        if dis < target_dis :
                            if ghost is not None :
                                if (i,j) != ghost:
                                    is_localized = True
                                    mini_idx.append((i,j))
                            else:
                                is_localized = True
                                mini_idx.append((i,j))
        else:
            for i in range(len(position_A_list)):
                dis = pu.get_l2_distance(position_A_list[i][0], position_B[0], \
                                            position_A_list[i][1], position_B[1])
                if dis < target_dis :
                    is_localized = True
                    mini_idx.append(i)
        return mini_idx, is_localized

    def check_around(self, pos):
        _, is_node = self.if_nearby(self.node_position_list, pos, target_dis=2)
        _, is_ghost = self.if_nearby(self.ghost_node_position, pos, target_dis=2)
        return not (is_node or is_ghost)

    def ghost_check(self):
        for j in range(len(self.node_position_list)):
            index, is_localized = self.if_nearby(self.ghost_node_position, self.node_position_list[j], target_dis=2)
            if is_localized:
                for i in range(len(index)):
                    self.ghost_mask[index[i][0],index[i][1]] = 0
        for i in range(self.ghost_node_position.shape[0]-1, -1, -1):
            for j in range(self.ghost_node_position.shape[1]-1, -1, -1):
                if self.ghost_mask[i,j] == 0:
                    continue
                else:
                    ghost_index, is_ghost_localized = self.if_nearby(self.ghost_node_position, self.ghost_node_position[i, j], ghost = (i,j), target_dis=0.5)
                    if is_ghost_localized:
                        for i in range(len(ghost_index)):
                            self.ghost_mask[ghost_index[i][0], ghost_index[i][1]] = 0
           

    def dijkstra(self, target, agent_id, length_only=False):
        source = self.last_localized_node_idx[agent_id]
        if length_only and source==target[agent_id]:
            return np.array([])
        matrix = self.A.copy()
        M = 1E100
        matrix[matrix==0] = M
        
        n = len(matrix)
        m = len(matrix[0])
        if source >= n or n != m:
            print('Error!')
            return
        found = [source]        
        cost = [M] * n         
        cost[source] = 0
        path = [[]]*n          
        path[source] = [source]
        target_path = None

        while len(found) < n:   
            min_value = M+1
            col = -1
            row = -1
            for f in found:    
                for i in [x for x in range(n) if x not in found]:   
                    if matrix[f][i] + cost[f] < min_value:  
                        min_value = matrix[f][i] + cost[f]  
                        row = f         
                        col = i
            if col == -1 or row == -1:  
                break
            found.append(col)           
            cost[col] = min_value       
            path[col] = path[row][:]    
            path[col].append(col)       
            if col == int(target[agent_id]):
                target_path = path[col]
        return  target_path

    def check_ghost_outside(self, map_in, explored_in, explored_all, ratio, resolution):
        if len(self.node_position_list) == self.num_init_nodes:
            pass
        elif np.any(ratio < 0.3) and (not self.flag_reset):
            self.last_ghost_mask = self.ghost_mask[:self.num_init_nodes].copy()
            self.ghost_mask[:self.num_init_nodes,:] = 0
            self.flag_reset = True
        elif np.all(ratio >= 0.3) and self.flag_reset:
            self.ghost_mask[:self.num_init_nodes] = self.last_ghost_mask
        ghost_pos = self.ghost_node_position
        world_pos = self.node_position_list
        for idx in range(ghost_pos.shape[0]):
            for idy in range(ghost_pos.shape[1]):
                if self.ghost_mask[idx,idy] == 0:
                    continue
                else:
                    curr_loc = world_pos[idx]
                    y, x = round(curr_loc[0]*100/resolution), round(curr_loc[1]*100/resolution)
                    yy, xx = round(ghost_pos[idx,idy][0]*100/resolution),round(ghost_pos[idx,idy][1]*100/resolution) 
                    dx = x-xx
                    dy = y-yy
                    ref = max(abs(dx), abs(dy))
                    step_x = dx/ref
                    step_y = dy/ref
                    if abs(step_x) > abs(step_y):
                        for i in range(0,abs(dx)+1):
                            tempx, tempy = round(xx+i*step_x), round(yy+i*step_y)
                            if explored_in[tempx, tempy-1] > 0.5 and explored_in[tempx, tempy] > 0.5 and explored_in[tempx, tempy+1] > 0.5 and \
                            explored_in[round(tempx+1), round(tempy-1)] > 0.5 and explored_in[round(tempx+1), round(tempy)] > 0.5 and explored_in[round(tempx+1), round(tempy+1)] > 0.5 and \
                            explored_in[round(tempx-1), round(tempy-1)] > 0.5 and explored_in[round(tempx-1), round(tempy)] > 0.5 and explored_in[round(tempx-1), round(tempy+1)] > 0.5:
                                break
                            elif map_in[tempx, tempy] > 0.2:
                                count = 0
                                for j in range(i+1,abs(dx)+1):
                                    if explored_all[round(xx+j*step_x)-1:round(xx+j*step_x)+2, round(yy+j*step_y)-1:round(yy+j*step_y)+2].sum()>4:
                                        count += 1
                                    if count > 0.6*(abs(dx)-i):
                                        self.ghost_mask[idx,idy] = 0
                                        break
                                break
                    else:
                        for i in range(0,abs(dy)+1):
                            tempx, tempy = round(xx+i*step_x), round(yy+i*step_y)
                            if explored_in[tempx, tempy-1] > 0.5 and explored_in[tempx, tempy] > 0.5 and explored_in[tempx, tempy+1] > 0.5 and \
                            explored_in[round(tempx+1), round(tempy-1)] > 0.5 and explored_in[round(tempx+1), round(tempy)] > 0.5 and explored_in[round(tempx+1), round(tempy+1)] > 0.5 and \
                            explored_in[round(tempx-1), round(tempy-1)] > 0.5 and explored_in[round(tempx-1), round(tempy)] > 0.5 and explored_in[round(tempx-1), round(tempy+1)] > 0.5:
                                break
                            elif map_in[tempx, tempy] > 0.2:
                                count = 0
                                for j in range(i+1,abs(dy)+1):
                                    if explored_all[round(xx+j*step_x)-1:round(xx+j*step_x)+2, round(yy+j*step_y)-1:round(yy+j*step_y)+2].sum()>4:
                                        count += 1
                                    if count > 0.6*(abs(dy)-i):
                                        self.ghost_mask[idx,idy] = 0
                                        break
                                break
    
  