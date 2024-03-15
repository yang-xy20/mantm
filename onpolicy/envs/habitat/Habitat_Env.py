import numpy as np
import gym
import onpolicy
from .exploration_env import Exploration_Env
import habitat
from habitat.config.default import get_config as cfg_env
from habitat_baselines.config.default import get_config as cfg_baseline
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from onpolicy.utils.multi_discrete import MultiDiscrete
from .utils.graph import Graph
import torch.nn as nn
import torch
import os

class MultiHabitatEnv(object):
    def __init__(self, args, rank, run_dir):
        self.all_args = args
        self.rank = rank
        self.run_dir = run_dir
        self.num_agents = args.num_agents
        self.use_merge_partial_reward = args.use_merge_partial_reward
        self.local_map_h = args.local_map_h
        self.local_map_w = args.local_map_w
        self.use_local = args.use_local
        self.use_centralized_V = args.use_centralized_V
        self.build_graph = args.build_graph
        self.add_ghost = args.add_ghost
        self.num_local_steps = args.num_local_steps
        self.ghost_node_size = args.ghost_node_size
        self.use_mgnn = args.use_mgnn
        self.rank = rank
        config_env, config_baseline, dataset = self.get_config(args, rank)
        self.env = Exploration_Env(
            args, config_env, config_baseline, dataset, run_dir, rank)
        self.config_env = config_env
        self.config_baseline = config_baseline
        self.dataset = dataset

        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / args.global_downscaling), \
            int(full_h / args.global_downscaling)
        global_observation_space = self.build_graph_global_obs()
        share_global_observation_space = global_observation_space.copy()
        
        if self.use_centralized_V:
            if self.use_local:
                share_global_observation_space['gt_map'] = gym.spaces.Box(
                    low=0, high=1, shape=(1, self.local_map_w, self.local_map_h), dtype='uint8')
            else:
                share_global_observation_space['gt_map'] = gym.spaces.Box(
                    low=0, high=1, shape=(1, local_w, local_h), dtype='uint8')
                        
        global_observation_space = gym.spaces.Dict(global_observation_space)
        share_global_observation_space = gym.spaces.Dict(share_global_observation_space)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for _ in range(self.num_agents):
            self.observation_space.append(global_observation_space)
            self.share_observation_space.append(share_global_observation_space)
            if self.use_mgnn:
                self.action_space.append(gym.spaces.Discrete(self.graph_memory_size*self.ghost_node_size)) 
            else:
                self.action_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32))
                

    def get_config(self, args, rank):
        config_env = cfg_env(config_paths=[onpolicy.__path__[0] + "/envs/habitat/habitat-lab/configs/" + args.task_config])
        
        config_env.defrost()
        config_env.DATASET.SPLIT = args.split
        if args.dataset == 'mp3d' or args.dataset == 'hm3d':
            config_env.DATASET.DATA_PATH = onpolicy.__path__[0] + "/envs/habitat/data/datasets/{}/{}/{}.json.gz".format(args.dataset, args.split,args.split)
        else:
            config_env.DATASET.DATA_PATH = onpolicy.__path__[0] + "/envs/habitat/data/datasets/pointnav/{}/v1/{}/{}.json.gz".format(args.dataset, args.split,args.split)
        
        config_env.freeze()

        scenes = PointNavDatasetV1.get_scenes_to_load(config_env.DATASET)
        
        if len(scenes) > 0:
            assert len(scenes) >= args.n_rollout_threads, (
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )
            scene_split_size = int(
                np.floor(len(scenes) / args.n_rollout_threads))

        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[rank *
                                                       scene_split_size: (rank + 1) * scene_split_size]
        config_env.DATASET.USE_SAME_SCENE = args.use_same_scene
        if args.use_same_scene:
            config_env.DATASET.CONTENT_SCENES = scenes[args.scene_id:args.scene_id+1]
        if args.use_selected_small_scenes:
            if args.dataset == 'hm3d':
                scene_num=[11, 23, 28, 34, 39, 41, 45, 47, 71, 77, 101] #9, 19, 36, 50, 64, 68, 69, 70, 73, 75, 82, 87, 88, 96, 98, 112, 46, 59
            else:
                scene_num=[8, 58, 27, 29, 26, 71, 12, 54, 57, 5]
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]
        if args.use_selected_middle_scenes:
            if args.dataset == 'hm3d':
                scene_num=[67, 89, 117, 165, 119] #10, 16, 17, 24, 38, 60, 78, 105, 109, 134, 144, 158, 158, 170, 172, 173, 18
            else:
                scene_num=[20, 16, 48, 22, 21, 43, 36, 61, 49]#40
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]
        if args.use_selected_large_scenes:
            if args.dataset == 'hm3d':
                scene_num=[278, 284] #54, 90, 94, 182, 201, 242, 274
            else:
                scene_num=[31, 70, 9, 47, 45]
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]
        if args.use_selected_overall_scenes:
            scene_num=[31, 70, 9, 47, 45, 20, 49, 48, 61, 43]
            config_env.DATASET.CONTENT_SCENES = scenes[scene_num[rank]:scene_num[rank]+1]

        if rank > (args.n_rollout_threads)/2 and args.n_rollout_threads > 5:
            gpu_id = 2
        else:
            gpu_id = 0 if torch.cuda.device_count() == 1 else 1

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
        config_env.TASK.NUM_AGENTS = self.num_agents

        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        config_env.SIMULATOR.NUM_AGENTS = self.num_agents
        config_env.SIMULATOR.SEED = rank * 5000 + args.seed
        config_env.SIMULATOR.USE_SAME_ROTATION = args.use_same_rotation
        config_env.SIMULATOR.USE_RANDOM_ROTATION = args.use_random_rotation
        config_env.SIMULATOR.USE_DIFFERENT_START_POS = args.use_different_start_pos
        config_env.SIMULATOR.USE_FIXED_START_POS = args.use_fixed_start_pos
        config_env.SIMULATOR.USE_FULL_RAND_STATE = args.use_full_rand_state
        config_env.SIMULATOR.CHANGE_AGENTS = (args.change_up_agents or args.change_down_agents)
        if args.use_same_rotation:
            config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/same_rot_state/seed{}/".format(args.seed)
        else:
            if args.use_random_rotation:
                config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/rand_rot_state/seed{}/".format(args.seed)
            else:
                config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/state/seed{}/".format(args.seed)

        config_env.SIMULATOR.AGENT.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]
        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]
        if self.build_graph:
            config_env = self.add_panoramic_camera(config_env)
        config_env.SIMULATOR.TURN_ANGLE = 10
        dataset = PointNavDatasetV1(config_env.DATASET)
        config_env.defrost()
        config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id

        print("Loading {}".format(config_env.SIMULATOR.SCENE))

        config_env.freeze()

        config_baseline = cfg_baseline()
      
        return config_env, config_baseline, dataset

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, reset_choose = True):
        obs, infos = self.env.reset()
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        if self.use_merge_partial_reward:
            rewards = 0.5 * np.expand_dims(np.array(infos['explored_merge_reward']), axis=1) + 0.5 * (np.expand_dims(np.array(infos['overlap_reward']), axis=1)+np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1))
        else:
            rewards = np.expand_dims(np.array(infos['overlap_reward'] ), axis=1)+np.expand_dims(np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_short_term_goal(self, inputs):
        outputs = self.env.get_short_term_goal(inputs)
        return outputs
    
    def get_runner_fmm_distance(self, data):
        outputs = self.env.get_runner_fmm_distance(data)
        return outputs

    def get_rrt_goals(self, inputs):
        outputs = self.env.get_rrt_goals(inputs)
        return outputs
    
    def prepare_direction_input(self, inputs):
        outputs = self.env.prepare_direction_input(inputs)
        return outputs
    
    def reset_scene(self, scene_id, num_agents, start_episode = None):
        config_env = self.config_env
        args = self.all_args
        config_env.defrost()
        scenes = PointNavDatasetV1.get_scenes_to_load(config_env.DATASET)
        config_env.DATASET.USE_SAME_SCENE = True
        config_env.DATASET.CONTENT_SCENES = [scenes[scene_id]]

        print("reset scene",  scenes[scene_id], "num agents", num_agents)

        config_env.SIMULATOR.NUM_AGENTS = num_agents
        config_env.SIMULATOR.SEED = np.random.randint(0,1000000000)
        
        if args.use_same_rotation:
            config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/same_rot_state/seed{}/".format(args.seed)
        else:
            if args.use_random_rotation:
                config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/rand_rot_state/seed{}/".format(args.seed)
            else:
                config_env.SIMULATOR.FIXED_MODEL_PATH = onpolicy.__path__[0] + "/envs/habitat/data/state/seed{}/".format(args.seed)
        
        dataset = PointNavDatasetV1(config_env.DATASET)
        config_env.defrost()

        config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id

        print("Loading {}".format(config_env.SIMULATOR.SCENE))

        config_env.freeze()

        config_baseline = cfg_baseline()

        start_episode = self.env.episode_no if start_episode is None else start_episode

        self.env.close()
        del self.env

        self.env = Exploration_Env(
            self.all_args, config_env, config_baseline, dataset, self.run_dir, start_episode = start_episode, first_time = False, rank = self.rank)
