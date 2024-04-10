from copy import deepcopy
from joblib import register_parallel_backend
import numpy as np
import math
import gym
from onpolicy.envs.habitat.utils.extractor import VisualEncoder
import onpolicy
from .exploration_env import Exploration_Env
import habitat
from habitat.config.default import get_config as cfg_env
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from onpolicy.utils.multi_discrete import MultiDiscrete
from .utils.graph import Graph, Merge_Graph
from .utils.mtsp import compute_kmeans, compute_frontier_plan
import torch.nn as nn
import torch
import os
from .Habitat_Env import MultiHabitatEnv
from torchvision import transforms
from PIL import Image
from .utils import pose as pu
import random
from icecream import ic
import time
from collections import deque
class GraphHabitatEnv(MultiHabitatEnv):
    def __init__(self, args, rank, run_dir):
        self.graph_memory_size = args.graph_memory_size
        self.feature_dim = args.feature_dim
        self.input_shape = (args.paro_frame_width, args.paro_frame_height)    
        self.num_agents = args.num_agents    
        self.graph = Graph(self.graph_memory_size, self.num_agents, self.input_shape, self.feature_dim)
        self.merge_graph = Merge_Graph(args,self.graph_memory_size, self.num_agents, self.input_shape, self.feature_dim)
        self.th = args.graph_th
        self.paro_hfov = args.paro_hfov
        self.paro_position_y = args.paro_position_y
        self.max_episode_length = args.max_episode_length
        self.use_id_embedding = args.use_id_embedding
        self.num_local_steps = args.num_local_steps
        self.add_ghost = args.add_ghost
        self.map_resolution = args.map_resolution
        self.use_merge = args.use_merge
        self.num_local_steps = args.num_local_steps
        self.use_restrict_graph = args.use_restrict_graph
        self.use_map_critic = args.use_map_critic
        self.ghost_node_size = args.ghost_node_size
        self.map_shape = (args.map_size_cm // args.map_resolution, args.map_size_cm // args.map_resolution)
        self.reset_all_memory(self.num_agents)
        self.log = None
        self.use_prediction = args.use_prediction
        self.resolution = args.map_resolution
        self.use_edge_info = args.use_edge_info
        self.use_mgnn = args.use_mgnn
        self.dis_gap = args.dis_gap
        self.episode_length = args.episode_length
        self.use_frontier_nodes = args.use_frontier_nodes
        self.max_frontier = args.max_frontier
        self.use_double_matching = args.use_double_matching
        self.use_single = args.use_single
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        self.local_w, self.local_h = int(full_w / args.global_downscaling), \
            int(full_h / args.global_downscaling)
        random.seed(rank)
        super().__init__(args, rank, run_dir)
        
    def reset(self, reset_choose = True):
        self.timesteps = 0
        self.imgs = [] if self.use_prediction and self.add_ghost else None
        self.positions = [] if self.use_prediction and self.add_ghost else None
        obs, infos = self.env.reset()
        return obs, infos

    def update_merge_graph(self, infos):
        curr_vis_embedding = infos['graph_curr_vis_embedding']
        if self.use_prediction:
            for agent in range(self.num_agents):
                self.imgs.append(curr_vis_embedding[agent])
                (x, y, o)  =  self.env.world_transform(agent, self.env.curr_loc[agent])
                self.positions.append([x,y,o])
        img_tensor = torch.cat((torch.tensor(infos['graph_panoramic_rgb'])/255.0, torch.tensor(infos['graph_panoramic_depth'])),3).permute(0,3,1,2)
        self.localize(curr_vis_embedding, infos['position'], infos['world_position'], infos['time'], infos['labels'], infos['ghost_loc'], infos['world_ghost_loc'],[False for _ in range(self.num_agents)], img_tensor)
        self.agent_last_pos = infos['world_position']
        self.add_node_flag = np.array([True for _ in range(self.num_agents)])
        global_memory_dict = self.get_global_memory()
        infos = self.update_infos(infos, global_memory_dict)
        self.convert_info()
       
        return infos

    def update_merge_step_graph(self, infos):
        curr_vis_embedding = infos['graph_curr_vis_embedding']
        img_tensor = torch.cat((torch.tensor(infos['graph_panoramic_rgb'])/255.0, torch.tensor(infos['graph_panoramic_depth'])),3).permute(0,3,1,2)
        self.localize(curr_vis_embedding, infos['position'], infos['world_position'], infos['time'], infos['labels'], infos['ghost_loc'], infos['world_ghost_loc'],[False for _ in range(self.num_agents)], img_tensor)
        
        if self.all_args.cut_ghost:
           
            if np.any(infos['add_node']):
                add_node = infos['add_node']                
                self.localize(curr_vis_embedding, infos['position'], infos['world_position'], infos['time'], infos['labels'], infos['ghost_loc'], infos['world_ghost_loc'],[False for _ in range(self.num_agents)], img_tensor, add_node=add_node)
            else:
                add_node= [False for _ in range(self.num_agents)]
            
            if infos['update'] is True:
                self.agent_last_pos = infos['world_position']
                
                self.merge_graph.check_ghost_outside(self.env.merge_map, self.env.merge_explored_map, self.env.merge_all_explored_map, self.env.ratio, self.resolution)
                self.merge_graph.ghost_check()
                self.add_node_flag = np.array([True for _ in range(self.num_agents)])
                add_node_check = 0
                
                for agent_id in range(self.num_agents):
                    if not add_node[agent_id]:
                       
                        if self.merge_graph.check_around(infos['world_position'][agent_id]):
                            add_node[agent_id] = True
                if np.any(add_node):
                    self.localize(curr_vis_embedding, infos['position'], infos['world_position'], infos['time'], infos['labels'], infos['ghost_loc'], infos['world_ghost_loc'],[False for _ in range(self.num_agents)], img_tensor, add_node=add_node)
                    self.merge_graph.check_ghost_outside(self.env.merge_map, self.env.merge_explored_map, self.env.merge_all_explored_map, self.env.ratio, self.resolution)
                    self.merge_graph.ghost_check()
                
              
                while self.merge_graph.ghost_mask.sum() < self.num_agents:
                    add_node = np.array([True for _ in range(self.num_agents)])
                    self.localize(curr_vis_embedding, infos['position'], infos['world_position'], infos['time'], infos['labels'], infos['ghost_loc'], infos['world_ghost_loc'],[False for _ in range(self.num_agents)], img_tensor, add_node=add_node)
                    if add_node_check == 1:
                        break
                    self.merge_graph.check_ghost_outside(self.env.merge_map, self.env.merge_explored_map, self.env.merge_all_explored_map, self.env.ratio, self.resolution)
                    self.merge_graph.ghost_check()
                    add_node_check += 1 
               
                
        
        global_memory_dict = self.get_global_memory()
        infos = self.update_infos(infos, global_memory_dict)
        self.convert_info()

       
        if self.use_merge_partial_reward:
            rewards = 0.5 * np.expand_dims(np.array(infos['explored_merge_reward']), axis=1) + 0.5 * (np.expand_dims(np.array(infos['overlap_reward']), axis=1)+np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1))
        
        else:
            rewards = np.expand_dims(np.array(infos['overlap_reward'] ), axis=1)+np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        
        
        return infos, rewards
    
    def step(self, actions):
        self.timesteps += 1
       
        obs, rewards, dones, infos = self.env.step(actions)
        
        return obs, rewards, dones, infos
        
    def reset_all_memory(self, num_agents=None):
        self.graph.reset(num_agents)
        self.merge_graph.reset()
        
    def add_panoramic_camera(self, task_config, normalize_depth=True):
        
        num_of_camera = 360//self.paro_hfov
        
        assert isinstance(num_of_camera, int)
        angles = [2 * np.pi * idx/ num_of_camera for idx in range(num_of_camera-1,-1,-1)]
        half = num_of_camera//2
        angles = angles[half:] + angles[:half]
        task_config.TASK.SENSORS += ['PANORAMIC_SENSOR', 'PANORAMIC_DEPTH_SENSOR']
        use_semantic = 'PANORAMIC_SEMANTIC_SENSOR' in task_config.TASK.SENSORS
        use_depth = 'PANORAMIC_DEPTH_SENSOR' in task_config.TASK.SENSORS
        
        sensors_with_ids = []
        sensors = []
        
        for camera_idx in range(num_of_camera):
            curr_angle = angles[camera_idx]
            if curr_angle > 3.14:
                curr_angle -= 2 * np.pi
            new_camera_config = task_config.SIMULATOR.RGB_SENSOR.clone()
            new_camera_config.WIDTH = self.input_shape[0]//num_of_camera
            new_camera_config.HEIGHT = self.input_shape[1]
            new_camera_config.HFOV = self.paro_hfov
            new_camera_config.POSITION = [0, self.paro_position_y, 0]
            new_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_camera_config.TYPE = "PanoramicPartRGBSensor"
            new_camera_config.ANGLE = "{}".format(camera_idx)
            task_config.SIMULATOR.update({'RGB_SENSOR_{}'.format(camera_idx): new_camera_config})
            sensors.append('RGB_SENSOR_{}'.format(camera_idx))
            
            if use_depth:
                new_depth_camera_config = task_config.SIMULATOR.DEPTH_SENSOR.clone()
                new_depth_camera_config.WIDTH = self.input_shape[0]//num_of_camera
                new_depth_camera_config.HEIGHT = self.input_shape[1]
                new_depth_camera_config.HFOV = self.paro_hfov
                new_depth_camera_config.POSITION = [0, self.paro_position_y, 0]
                new_depth_camera_config.ORIENTATION = [0, curr_angle, 0]
                new_depth_camera_config.TYPE = "PanoramicPartDepthSensor"
                new_depth_camera_config.ANGLE = "{}".format(camera_idx)
                new_depth_camera_config.NORMALIZE_DEPTH = normalize_depth
                task_config.SIMULATOR.update({'DEPTH_SENSOR_{}'.format(camera_idx): new_depth_camera_config})
                sensors.append('DEPTH_SENSOR_{}'.format(camera_idx))
            if use_semantic:
                new_semantic_camera_config = task_config.SIMULATOR.SEMANTIC_SENSOR.clone()
                new_semantic_camera_config.TYPE = "PanoramicPartSemanticSensor"
                new_semantic_camera_config.ORIENTATION = [0, curr_angle, 0]
                new_semantic_camera_config.ANGLE = "{}".format(camera_idx)
                task_config.SIMULATOR.update({'SEMANTIC_SENSOR_{}'.format(camera_idx): new_semantic_camera_config})
                sensors.append('SEMANTIC_SENSOR_{}'.format(camera_idx))        
        
        task_config.SIMULATOR.AGENT.SENSORS += sensors
        
        sensor_dict = {'TYPE': 'PanoramicRGBSensor', 'WIDTH': task_config.SIMULATOR.RGB_SENSOR.HEIGHT * 4,
                    'HEIGHT': task_config.SIMULATOR.RGB_SENSOR.HEIGHT, 'NUM_CAMERA': num_of_camera,'AGENT_ID': str(id)
                    }
        task_config.TASK['PANORAMIC_SENSOR'] = habitat.Config()
        task_config.TASK['PANORAMIC_SENSOR'].update(sensor_dict)
        sensors_with_ids.append('PANORAMIC_SENSOR')
        if use_depth:
            task_config.TASK['PANORAMIC_DEPTH_SENSOR'] = task_config.TASK['PANORAMIC_SENSOR'].clone()
            task_config.TASK['PANORAMIC_DEPTH_SENSOR'].TYPE = 'PanoramicDepthSensor'
            task_config.TASK['PANORAMIC_DEPTH_SENSOR'].NORMALIZE_DEPTH = True
            task_config.TASK['PANORAMIC_DEPTH_SENSOR'].MIN_DEPTH = 0.0
            task_config.TASK['PANORAMIC_DEPTH_SENSOR'].MAX_DEPTH = 10.0
            sensors_with_ids.append('PANORAMIC_DEPTH_SENSOR')
        if use_semantic:
            task_config.TASK['PANORAMIC_SEMANTIC_SENSOR'] = task_config.TASK['PANORAMIC_SENSOR'].clone()
            task_config.TASK['PANORAMIC_SEMANTIC_SENSOR'].TYPE = 'PanoramicSemanticSensor'
            sensors_with_ids.append('PANORAMIC_SEMANTIC_SENSOR')
        
        return task_config
    
    
        
    def is_close(self, embed_a, embed_b, return_prob=False):
        logits = np.matmul(np.expand_dims(embed_a,1), np.expand_dims(np.array(embed_b),2)).squeeze(2).squeeze(1)
        close = (logits > self.th)
        if return_prob: return close, logits
        else: return close
        
    # assume memory index == node index
    def localize(self, new_embedding, position, world_position, time, labels, ghost_position, world_ghost_position, done_list, raw_img=None, add_node=False):
        # The position is only used for visualizations.
        time = np.array(time)[:,0]

        if np.any(add_node): 
            for agent_id in range(self.num_agents):
                if add_node[agent_id]:
                    new_node_idx = self.merge_graph.num_node()
                    self.merge_graph.add_node(new_node_idx, agent_id, new_embedding[agent_id], time[agent_id], world_position[agent_id])
              
                    dis = 0
                    self.merge_graph.add_edge(new_node_idx, self.merge_graph.last_localized_node_idx[agent_id], dis)
                    self.merge_graph.record_localized_state(new_node_idx, agent_id, new_embedding[agent_id])
                    if self.add_ghost:
                       
                        self.merge_graph.add_ghost_node(agent_id, labels[agent_id], world_ghost_position[agent_id], self.imgs, self.positions, raw_img)
        else:
            done = np.where(done_list)[0]
            if time.sum() == 0:
                self.merge_graph.reset_at()
                for b in range(self.num_agents):
                    self.graph.reset_at(b)
                    self.graph.initialize_graph(b, new_embedding, position)
                    if b != 0:
                        logits = np.matmul(self.merge_graph.graph_memory, new_embedding[b]).squeeze()
                        close = (logits > self.th)[0]
                        if close.any():
                            idx = np.where(close==True)[0][0]
                            self.merge_graph.record_localized_state(idx, b, self.merge_graph.graph_memory[idx])
                            new_embedding[b] = new_embedding[idx]
                            continue
                        else:
                            new_node_idx = self.merge_graph.num_node()
                            self.merge_graph.add_node(new_node_idx, b, new_embedding[b], time[b], world_position[b])
                            self.merge_graph.record_localized_state(new_node_idx, b, new_embedding[b])
                    else:
                        self.merge_graph.initialize_graph(b, new_embedding[b], world_position[b])
                    if self.add_ghost:
                       
                        self.merge_graph.add_ghost_node(b, labels[b], world_ghost_position[b], self.imgs, self.positions, raw_img)
                self.merge_graph.num_init_nodes = len(self.merge_graph.node_position_list)

    def single_localize(self, new_embedding, position, world_position, time, labels, ghost_position, world_ghost_position, done_list, raw_img=None, add_node=False):
        # The position is only used for visualizations.
        time = np.array(time)[:,0]

        if np.any(add_node): 
            for agent_id in range(self.num_agents):
                if add_node[agent_id]:
                    new_node_idx = deque(maxlen = 1)
                    exec('new_node_idx.append(self.merge_graph_agent_{}.num_node())'.format(agent_id))
                    exec('self.merge_graph_agent_{}.add_node(new_node_idx[-1], agent_id, new_embedding[agent_id], time[agent_id], world_position[agent_id])'.format(agent_id))
                    
                    dis = 0
                    exec('self.merge_graph_agent_{}.add_edge(new_node_idx[-1], self.merge_graph.last_localized_node_idx[agent_id], dis)'.format(agent_id))
                    exec('self.merge_graph_agent_{}.record_localized_state(new_node_idx[-1], agent_id, new_embedding[agent_id])'.format(agent_id))
                    if self.add_ghost:
                       
                        exec('self.merge_graph_agent_{}.add_ghost_node(agent_id, labels[agent_id], world_ghost_position[agent_id], self.imgs, self.positions, raw_img)'.format(agent_id))
        else:
            done = np.where(done_list)[0]
            if time.sum() == 0:
                for b in range(self.num_agents):
                    exec('self.merge_graph_agent_{}.reset_at()'.format(b))
                    exec('self.merge_graph_agent_{}.initialize_graph(b, new_embedding[b], world_position[b])'.format(b))
                    if self.add_ghost:
                      
                        exec('self.merge_graph_agent_{}.add_ghost_node(b, labels[b], world_ghost_position[b], self.imgs, self.positions, raw_img)'.format(b))
                exec('self.merge_graph_agent_{}.num_init_nodes = len(self.merge_graph_agent_{}.node_position_list)'.format(b,b))

    def convert_info(self):
        self.env.node_list = self.graph.node_position_list
        self.env.affinity = self.graph.A
       
        self.env.merge_node_list = self.merge_graph.node_position_list
        if self.use_edge_info is not None:
            temp_A = np.zeros_like(self.merge_graph.A)
            temp_A[self.merge_graph.A > 0] = 1
            self.env.merge_affinity = temp_A
        else:
            self.env.merge_affinity = self.merge_graph.A
        self.env.merge_ghost_node = self.merge_graph.ghost_node_position
        self.env.merge_ghost_mask = self.merge_graph.ghost_mask
       
        self.env.current_index = self.merge_graph.last_localized_node_idx
        self.env.last_localized_node_idx = self.merge_graph.last_localized_node_idx
        self.env.cur_node_num = self.merge_graph.num_node()
        self.env.graph_creat_id = self.merge_graph.graph_creat_id
       

    def update_infos(self, infos, global_memory_dict):
        # add memory to obs
        for key in self.g_obs_space.keys():
            if key in ['graph_panoramic_rgb','agent_graph_node_dis','graph_agent_dis','graph_panoramic_depth','graph_time','graph_prev_actions', 'graph_ghost_valid_mask',\
            'graph_prev_goal','graph_agent_id','agent_world_pos','graph_curr_vis_embedding','graph_agent_mask','global_merge_goal','global_merge_obs',\
            'graph_last_ghost_node_position','graph_merge_frontier_mask','graph_last_agent_world_pos','last_graph_dis','graph_last_agent_dis',\
            'graph_last_pos_mask','graph_last_ghost_node_position','graph_last_node_position','graph_merge_frontier_mask','graph_last_agent_world_pos','last_graph_dis','graph_last_agent_dis',\
            'graph_last_pos_mask']:
                pass
           
            elif key == "graph_id_embed":
                infos[key] = np.array([[1*a] for a in range(self.num_agents)])
            elif key == 'merge_node_pos':
                infos[key] = np.expand_dims(global_memory_dict[key], axis=0).repeat(self.num_agents, 0)
           
            else:
                infos[key] = global_memory_dict[key]
        return infos
    
    def get_global_memory(self, mode='feature'):
        global_memory_dict = { }
        
        if self.use_merge:
            global_memory_dict['graph_ghost_node_position'] = self.merge_graph.ghost_node_position
            if not self.use_mgnn:
                global_memory_dict['graph_merge_global_memory'] = self.merge_graph.graph_memory
                global_memory_dict['graph_merge_global_act_memory'] = self.merge_graph.graph_act_memory
                global_memory_dict['graph_merge_global_mask'] = self.merge_graph.graph_mask
                global_memory_dict['graph_merge_global_A'] = self.merge_graph.A
                global_memory_dict['graph_merge_global_D'] = self.merge_graph.D
                global_memory_dict['graph_merge_localized_idx'] = np.expand_dims(self.merge_graph.last_localized_node_idx,1)
                global_memory_dict['graph_merge_global_time'] = self.merge_graph.graph_time
                global_memory_dict['merge_graph_id_trace'] = self.merge_graph.graph_id_trace
                global_memory_dict['merge_graph_num_node'] = [self.node_max_num()]
            if self.use_edge_info == 'learned':
                temp = np.zeros(self.merge_graph.graph_mask.shape)
                temp[self.merge_graph.graph_mask==1] = np.array([pos[2] for pos in self.merge_graph.node_position_list])
                global_memory_dict['merge_node_pos'] = temp
            if self.add_ghost:
               
                global_memory_dict['graph_merge_ghost_feature'] = self.merge_graph.ghost_node_feature
               
                global_memory_dict['graph_merge_ghost_mask'] = self.merge_graph.ghost_mask
            if self.use_double_matching:
                global_memory_dict['graph_node_pos'] = self.merge_graph.node_position_list
       
        if self.use_single:
            global_memory_dict['graph_global_memory']= self.graph.graph_memory
            global_memory_dict['graph_global_act_memory']= self.graph.graph_act_memory
            global_memory_dict['graph_global_mask']= self.graph.graph_mask
            global_memory_dict['graph_global_A']= self.graph.A
            global_memory_dict['graph_global_time']= self.graph.graph_time
            global_memory_dict['graph_id_trace'] = self.graph.graph_id_trace
            global_memory_dict['graph_localized_idx'] = np.expand_dims(self.graph.last_localized_node_idx,1)
            global_memory_dict['graph_num_node'] = [[self.graph.num_node(agent_id)] for agent_id in range(self.num_agents)]

        return global_memory_dict
    
    
    def build_graph_global_obs(self):
        self.g_obs_space = {}
        if self.use_map_critic:
            self.g_obs_space['global_merge_obs'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(6, self.local_w, self.local_h), dtype=np.float32)
            self.g_obs_space['global_merge_goal'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(2, self.local_w, self.local_h), dtype=np.float32)

        if self.use_single:
            self.g_obs_space['graph_global_memory'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.num_agents* self.graph_memory_size, self.feature_dim), dtype=np.float32)
            self.g_obs_space['graph_global_mask'] = gym.spaces.Box(
                        low=-np.Inf, high=np.Inf, shape=(self.num_agents* self.graph_memory_size,), dtype=np.int32)
            self.g_obs_space['graph_global_A'] = gym.spaces.Box(
                        low=-np.Inf, high=np.Inf, shape=(self.num_agents* self.graph_memory_size, self.graph_memory_size), dtype=np.int32)
            self.g_obs_space['graph_global_time'] = gym.spaces.Box(
                        low=-np.Inf, high=np.Inf, shape=(self.num_agents* self.graph_memory_size,), dtype=np.int32)
            self.g_obs_space['graph_localized_idx'] = gym.spaces.Box(
                        low=-np.Inf, high=np.Inf, shape=(self.num_agents* 1,), dtype=np.int32)
        if self.use_merge:
            if self.use_double_matching:
                self.g_obs_space['graph_last_node_position'] = gym.spaces.Box( low=-np.Inf, high=np.Inf, shape=(self.episode_length*self.num_agents, 4), dtype = np.float32)
                self.g_obs_space['agent_graph_node_dis'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.num_agents, self.graph_memory_size, 1), dtype=np.float32)
                self.g_obs_space['graph_node_pos'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size, 4), dtype=np.float32)
            if self.use_frontier_nodes:
                self.g_obs_space['graph_ghost_node_position'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.max_frontier, 4), dtype=np.float32)
            else:
                self.g_obs_space['graph_ghost_node_position'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size, self.ghost_node_size, 4), dtype=np.float32)
            self.g_obs_space['agent_world_pos'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.num_agents,4), dtype=np.int32)
            self.g_obs_space['graph_last_ghost_node_position'] = gym.spaces.Box( low=-np.Inf, high=np.Inf, shape=(self.episode_length*self.num_agents, 4), dtype = np.float32)
            self.g_obs_space['graph_last_agent_world_pos'] = gym.spaces.Box( low=-np.Inf, high=np.Inf, shape=(self.episode_length*self.num_agents, 4), dtype = np.float32)
            if self.use_mgnn:
               
               
                self.g_obs_space['graph_last_pos_mask'] = gym.spaces.Box(
                            low=0, high=np.Inf, shape=(self.episode_length*self.num_agents, 1), dtype=np.int32)
                if self.use_frontier_nodes:
                    self.g_obs_space['graph_merge_frontier_mask'] = gym.spaces.Box(
                                    low=-np.Inf, high=np.Inf, shape=(self.max_frontier, ),dtype = np.float32)
                if self.use_frontier_nodes:
                   
                    self.g_obs_space['graph_agent_dis'] = gym.spaces.Box(
                                    low=-np.Inf, high=np.Inf, shape=(self.num_agents, self.max_frontier, 1), dtype=np.float32)
                else:
                  
                    self.g_obs_space['graph_agent_dis'] = gym.spaces.Box(
                                    low=-np.Inf, high=np.Inf, shape=(self.num_agents, self.graph_memory_size*self.ghost_node_size, 1), dtype=np.float32)
              
            else:
                self.g_obs_space['graph_ghost_valid_mask'] = gym.spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size*self.ghost_node_size, self.graph_memory_size*self.ghost_node_size), dtype=np.float32)
                self.g_obs_space['graph_merge_global_memory'] = gym.spaces.Box(
                        low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size, self.feature_dim), dtype=np.float32)
                self.g_obs_space['graph_merge_global_mask'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size,), dtype=np.int32)
                self.g_obs_space['graph_merge_global_A'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size, self.graph_memory_size), dtype=np.int32)
                self.g_obs_space['graph_merge_global_time'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size,), dtype=np.int32)        
                self.g_obs_space['graph_merge_localized_idx'] = gym.spaces.Box(
                                low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.int32)
            if self.use_edge_info == 'learned':
                self.g_obs_space['merge_node_pos'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size,), dtype=np.int32)
          
            if self.add_ghost:
                self.g_obs_space['graph_merge_ghost_feature'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.graph_memory_size,self.ghost_node_size, self.feature_dim), dtype=np.float32)
               
                self.g_obs_space['graph_merge_ghost_mask'] = gym.spaces.Box(
                            low=0, high=np.Inf, shape=(self.graph_memory_size,self.ghost_node_size), dtype=np.int32)
       
        self.g_obs_space['graph_panoramic_rgb'] = gym.spaces.Box(
                        low=0, high=256, shape=(self.num_agents, self.input_shape[1], self.input_shape[0], 3), dtype=np.float32)
        self.g_obs_space['graph_panoramic_depth'] = gym.spaces.Box(
                        low=0, high=256, shape=(self.num_agents, self.input_shape[1], self.input_shape[0], 1), dtype=np.float32)
        if not self.use_mgnn:
            self.g_obs_space['graph_time'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.int32)
            self.g_obs_space['graph_prev_actions'] = gym.spaces.Box(
                            low=-np.Inf, high=np.Inf, shape=(self.num_local_steps,), dtype=np.int32)
       
      
       
        return self.g_obs_space
    
    def store_goal(self, goal):
        self.env.store_goal(goal)

    def get_graph_waypoint(self, global_goal):
        goal = np.zeros((self.num_agents, 2), np.int32)
        
        for agent_id in range(self.num_agents):
            (goal_x, goal_y, goal_o) = self.merge_graph.get_positions(int(global_goal[agent_id]))
            (start_x, start_y, start_o)  =  self.env.world_transform(agent_id, self.env.curr_loc[agent_id])
            if int(global_goal[agent_id]) == self.merge_graph.last_localized_node_idx[agent_id]:# or \
                ghost_node_list = np.argwhere(self.merge_graph.ghost_node_link == self.merge_graph.last_localized_node_idx[agent_id])
                if (len(ghost_node_list) is 0):
                    next_target_goal = self.merge_graph.last_localized_node_idx[agent_id]
                    (next_x, next_y, next_o) = self.merge_graph.get_positions(next_target_goal)
                else:
                    idx = random.randint(0,len(ghost_node_list[0])-1)
                    random_ghost = ghost_node_list[0,idx]
                    
                    (next_x, next_y, next_o) = self.merge_graph.get_ghost_positions(random_ghost)
            else:
                target_path = self.merge_graph.dijkstra(global_goal, agent_id)
                if target_path is None:
                    next_target_goal = self.merge_graph.last_localized_node_idx[agent_id]
                else:
                    next_target_goal = target_path[1]
                (next_x, next_y, next_o) = self.merge_graph.get_positions(next_target_goal)
            
            
            goal[agent_id] = [int(next_x * 100.0/self.map_resolution),
                        int(next_y * 100.0/self.map_resolution)]
        return goal, self.has_node
    
    def node_max_num(self):
        num_node = self.merge_graph.num_node()
        return num_node
    
    def get_valid_num(self, inp):
        if inp is None:
            return len(np.unique(self.merge_graph.ghost_node_link))
        else:
            return [np.unique(self.merge_graph.ghost_node_link)[math.floor(inp[i])] for i in range(len(inp))]

    def get_valid_index(self):
        return self.merge_graph.ghost_mask

    def get_graph_frontier(self):
        same_goal = len(np.unique(self.merge_graph.ghost_node_link)) < self.num_agents
        max_node_num = np.where(self.merge_graph.graph_mask==1)[0][-1] + 1
        ghost_score = np.zeros(max_node_num)
        temp_ghost = np.bincount(self.merge_graph.ghost_node_link)
        ghost_score[0:temp_ghost.shape[0]] = temp_ghost
        ghost_score[ghost_score==0] = -np.inf
        score = np.zeros((self.num_agents, max_node_num))
        for id in range(max_node_num):
            if self.merge_graph.graph_mask[id] == 1:
                for agent_id in range(self.num_agents):
                    score[agent_id, id] = len(self.merge_graph.dijkstra([id]*self.num_agents, agent_id, length_only=True))
            else:
                for agent_id in range(self.num_agents):
                    score[agent_id, id] = np.inf
        score = np.expand_dims(ghost_score, axis=0).repeat(self.num_agents, axis=0) - score / max_node_num
        max_score = -np.inf
        #TODO support more than 2 agents:dfs
        final_ans = []
        for i in range(max_node_num):
            for j in range(max_node_num):
                if not same_goal and i==j:
                    continue
                else:
                    s = score[0, i]+score[1,j]
                    if s > max_score:
                        max_score = s
                        final_ans = [i,j]

        
        return np.array(final_ans)

    def get_ghost_index(self, global_goal, ghost_goal):
        random_ghost = np.zeros((self.num_agents), np.int32)
        for agent_id in range(self.num_agents):
            ghost_node_list = np.argwhere(np.array(self.merge_graph.ghost_node_link) == int(global_goal[agent_id]))
            idx = random.randint(0,len(ghost_node_list)-1)
            random_ghost[agent_id] = ghost_node_list[idx]
        return random_ghost

    def get_goal_position(self, global_goal):
        self.valid_ghost_position = np.zeros((1000,3), np.float)
        self.valid_ghost_single_position = None
        goal = np.zeros((self.num_agents, 1, 2), np.int32)
        for agent_id in range(self.num_agents):
            for i in range(goal.shape[1]):
                (goal_x, goal_y, goal_o) = self.merge_graph.get_ghost_positions(int(global_goal[agent_id][i]), agent_id)
                goal[agent_id, i] = [int(goal_x * 100.0/self.map_resolution),
                        int(goal_y * 100.0/self.map_resolution)]
               
        valid_ghost_position = self.merge_graph.get_all_ghost_positions()
        self.valid_ghost_position[:valid_ghost_position.shape[0]] = valid_ghost_position * 100.0 / self.map_resolution
      
        return goal, self.valid_ghost_position, self.valid_ghost_single_position

    def compute_mtsp(self):
        ghost_node_pos = self.merge_graph.ghost_node_position[self.merge_graph.ghost_mask == 1]
        agent_pos = np.array([self.env.world_transform(agent_id, self.env.curr_loc[agent_id]) for agent_id in range(self.num_agents)])
        if ghost_node_pos.shape[0] >= 3*self.num_agents:
            _, centers = compute_kmeans(ghost_node_pos[:, :2], 3*self.num_agents)
        else:
            centers = ghost_node_pos[:, :2]
        selected_data, _ = compute_kmeans(centers, self.num_agents, agent_pos[:, :2]) 
        path = []
        for agent_id in range(self.num_agents):
            path = compute_frontier_plan(selected_data[agent_id])
            selected_data[agent_id] = selected_data[agent_id][path]
        return selected_data 