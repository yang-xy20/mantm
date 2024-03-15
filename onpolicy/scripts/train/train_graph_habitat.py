#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
from collections import deque

import torch

from onpolicy.config import get_config
from onpolicy.envs.habitat.Habitat_Env import MultiHabitatEnv
from onpolicy.envs.habitat.Graph_Habitat_Env import GraphHabitatEnv

from onpolicy.envs.env_wrappers import InfoSubprocVecEnv, InfoDummyVecEnv


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Habitat":
                if all_args.build_graph:
                    env = GraphHabitatEnv(args=all_args,
                                        rank=rank,
                                        run_dir=run_dir
                                        )
                else:
                    env = MultiHabitatEnv(args=all_args,
                                        rank=rank,
                                        run_dir=run_dir
                                        )
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    else:
        return InfoSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Habitat":
                env = MultiHabitatEnv(args=all_args,
                                      rank=rank,
                                      run_dir=run_dir
                                      )
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    else:
        return InfoSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='pointnav', help="Which scenario to run on")

    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")

    parser.add_argument('--train_local', action='store_true',
                        default=False, help="""0: Do not train the Local Policy
                                1: Train the Local Policy (default: 1)""")
    parser.add_argument('--train_slam', action='store_true',
                        default=False, help="""0: Do not train the Neural SLAM Module
                                1: Train the Neural SLAM Module (default: 1)""")

    parser.add_argument('--load_slam', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('--load_local', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('--direct_action_dir', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")

    # visual params
    parser.add_argument("--render_merge", action='store_false', default=True,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument('--save_trajectory_data', action='store_true', default=False)
    parser.add_argument('--use_same_scene', action='store_true', default=False)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument('--use_selected_small_scenes', action='store_true', default=False)
    parser.add_argument('--use_selected_middle_scenes', action='store_true', default=False)
    parser.add_argument('--use_selected_large_scenes', action='store_true', default=False)
    parser.add_argument('--use_selected_overall_scenes', action='store_true', default=False)
    
    # reward params
    parser.add_argument('--use_restrict_map',action='store_true', default=False)
    parser.add_argument('--use_time_penalty',action='store_true', default=False)
    parser.add_argument('--use_repeat_penalty',action='store_true', default=False)
    parser.add_argument('--use_complete_reward',action='store_true', default=False)
    parser.add_argument('--use_intrinsic_reward', action='store_true', default=False)
    parser.add_argument('--use_delta_reward', action='store_true', default=False)
    parser.add_argument('--use_partial_reward', action='store_true', default=False)
    parser.add_argument('--use_merge_partial_reward', action='store_true', default=False)
    parser.add_argument('--use_competitive_reward', action='store_true', default=False)
    parser.add_argument('--use_overlap_penalty', action='store_true', default=False)
    parser.add_argument('--use_circular_overlap_penalty', action='store_true', default=False)
    parser.add_argument('--use_pos_penalty', action='store_true', default=False)
    
    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=256,
                        help='Frame width (default:84)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=256,
                        help='Frame height (default:84)')
    parser.add_argument('-fw', '--frame_width', type=int, default=128,
                        help='Frame width (default:84)')
    parser.add_argument('-fh', '--frame_height', type=int, default=128,
                        help='Frame height (default:84)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=1000,
                        help="""Maximum episode length in seconds for
                                Doom (default: 180)""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/pointnav_gibson.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--dataset", type=str, default="gibson")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('-na', '--noisy_actions', type=int, default=1)
    parser.add_argument('-no', '--noisy_odometry', type=int, default=1)
    parser.add_argument('--camera_height', type=float, default=1.25,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=90.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--map_threshold', type=float, default=0.0,
                        help="lower than map_threshold will be setted to zero")
    parser.add_argument('--explored_ratio_down_threshold', type=float, default=0.9,
                        help="explored_ratio_down_threshold")
    parser.add_argument('--explored_ratio_up_threshold', type=float, default=0.95,
                        help="explored_ratio_up_threshold")
    parser.add_argument('--randomize_env_every', type=int, default=1000,
                        help="randomize scene in a thread every k episodes")
    parser.add_argument('--local_map_w', type=int, default=120,
                        help="local_map_width")
    parser.add_argument('--local_map_h', type=int, default=120,
                        help="local_map_height")
    parser.add_argument('--use_different_start_pos', action='store_true',
                        default=False, help="by default True, use random agent position at the initialization")
    parser.add_argument('--use_same_rotation', action='store_true',
                        default=False, help="by default True, use fixed agent rotation at the initialization")
    parser.add_argument('--use_random_rotation', action='store_true',
                        default=False, help="by default True, use random agent rotation at the initialization")
    parser.add_argument('--use_abs_orientation', action='store_true',
                        default=False, help="by default True, use abs orientation at the initialization")
    parser.add_argument('--use_center', action='store_true',
                        default=False, help="by default True, use agent center point as input")
    parser.add_argument('--use_resnet', action='store_true',
                        default=False, help="by default True, use resnet as global network")
    parser.add_argument('--use_vit', action='store_true',
                        default=False, help="by default True, use vit as global network")
    parser.add_argument('--cnn_use_attn', action = 'store_true',
                        default=False, help="by default True, add attention layer to the end of cnn network")
    parser.add_argument('--cnn_use_transformer', action = 'store_true',
                        default=False, help="by default True, add transformer to the end of cnn network")
    parser.add_argument('--use_fixed_start_pos', action='store_true',
                        default=False, help="by default True, use random agent position at the initialization")
    parser.add_argument('--use_full_rand_state', action='store_true',
                        default=False, help="by default True, use random agent position at the initialization")
    parser.add_argument('--use_rect', action='store_true',
                        default=False, help="by default True, use rectangular sim map")
    parser.add_argument('--use_max', action='store_true',
                        default=False, help="by default True, use maximun map or use sum map")   
    parser.add_argument('--use_orientation', action='store_true',
                        default=False, help="by default True, use agent orientation info")
    parser.add_argument('--use_vector_agent_id', action='store_true',
                        default=False, help="by default True, use fc net")
    parser.add_argument('--use_cnn_agent_id', action='store_true',
                        default=False, help="by default True, use fc net")
    parser.add_argument('--use_own', action='store_true',
                        default=False, help="by default True, use own vector cnn")
    parser.add_argument('--use_merge_mapper', action='store_true',
                        default=False, help="by default True, use gt merge_mapper")
    parser.add_argument('--use_new_trace', action='store_true',
                        default=False, help="by default True, the trace channel only has last goal steps' trace")
    parser.add_argument('--use_weight_trace', action='store_true',
                        default=False, help="by default True, use_weight_trace")
    parser.add_argument('--decay_weight', type=float, default=0.9,
                        help="decay_weight of new weight trace")
    parser.add_argument('--use_merge_goal', action='store_true',
                        default=False, help="by default True, use_merge_goal")
    parser.add_argument('--use_goal', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_single_agent_trace', action='store_true',
                        default=False, help="by default True, use_single_agent_weight_trace")
    parser.add_argument('--use_local_single_map', action='store_true',
                        default=False, help="by default True, use_single_agent_weight_trace")
    parser.add_argument('--use_single_merge', action='store_true',
                        default=False, help="by default True, use_single_agent_weight_trace")
    parser.add_argument('--use_local', action='store_true',
                        default=False, help="by default True, use_local_map")
    parser.add_argument('--use_seperated_cnn_model', action='store_true',
                        default=False, help="by default True, use_seperated_cnn_model")
    parser.add_argument('--grid_pos', action='store_true',
                        default=False, help="by default True, use grid_pos")
    parser.add_argument('--grid_agent_id', action='store_true',
                        default=False, help="by default True, use grid_agent_id")
    parser.add_argument('--use_agent_id', action='store_true',
                        default=False, help="by default True, use agent_id")
    parser.add_argument('--use_self_grid_pos', action='store_true',
                        default=False, help="by default True, use agent_id")
    parser.add_argument('--grid_last_goal', action='store_true',
                        default=False, help="by default True, use grid_last_goal")
    parser.add_argument('--proj_frontier', action='store_true',
                        default=False, help="by default True, restrict goals to frontiers")
    parser.add_argument('--use_map_agent_id', action='store_true',
                        default=False, help="by default True, use_map_agent_id")
    parser.add_argument('--use_single', action='store_true',
                        default=False, help="by default True, use_single")
    parser.add_argument('--agent_invariant', action='store_true',
                        default=False, help="by default True, ")
    parser.add_argument('--invariant_type', type=str, default = "attn_sum", choices = ["attn_sum", "split_attn", "mean", "alter_attn", "attn_rnn", "attn_N"])
    parser.add_argument('--multi_layer_cross_attn', default=False, action='store_true')
    parser.add_argument('--change_up_agents', action='store_true',
                        default=False, help="by default True, change_agents_num in the middle of the episode")
    parser.add_argument('--change_down_agents', action='store_true',
                        default=False, help="by default True, change_agents_num in the middle of the episode")
    parser.add_argument('--second_stage_agents', type=int, default=2,
                        help="""Number of second_stage_agents""")
    parser.add_argument('--first_stage_agents', type=int, default=2,
                        help="""Number of first_stage_agents""")
    parser.add_argument('--use_explicit_id', action='store_true',
                        default=False, help="by default True, use_random_id")
    parser.add_argument('--use_pos_embedding', action='store_true',
                        default=False, help="by default True, use_pos_embedding")
    parser.add_argument('--use_id_embedding', action='store_true',
                        default=False, help="by default True, use_id_embedding")
    parser.add_argument('--norm_sum', action='store_true',
                        default=False, help="by default True, norm_sum")
    parser.add_argument('--use_self_attn', action='store_true',
                        default=False, help="by default True, use_self_attn")
    parser.add_argument('--use_intra_attn', action='store_true',
                        default=False, help="by default True, use_intra_attn")
    parser.add_argument('--attn_depth', type=int, default=2,
                        help="""attn_depth""")
    parser.add_argument('--add_grid_pos', action='store_true',
                        default=False, help="by default True, add_grid_pos")
    parser.add_argument('--use_grid_pos_attn', action='store_true',
                        default=False, help="by default True, use_grid_pos_attn")
    parser.add_argument('--take_self', action='store_true',
                        default=False, help="by default True, take_self")
    parser.add_argument('--use_flatten_attn', action='store_true',
                        default=False, help="by default True, use_flatten_attn")
                        
    
    # Local Policy
    parser.add_argument('--local_lr', type=float, default=0.0001)
    parser.add_argument('--local_opti_eps', type=float, default=1e-5)
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local can
                            perform between each global instruction""")
    parser.add_argument('--local_hidden_size', type=int, default=512,
                        help='local_hidden_size')
    parser.add_argument('--short_goal_dist', type=int, default=1,
                        help="""Maximum distance between the agent
                                and the short term goal""")
    parser.add_argument('--local_policy_update_freq', type=int, default=5)
    parser.add_argument('--use_local_recurrent_policy', type=int, default=1,
                        help='use a recurrent local policy')
    parser.add_argument('--use_local_deterministic', type=int, default=0,
                        help="use classical deterministic local policy")

    # Neural SLAM Module
    parser.add_argument('--slam_lr', type=float, default=0.0001)
    parser.add_argument('--slam_opti_eps', type=float, default=1e-5)
    parser.add_argument('--use_pose_estimation', type=int, default=2)
    parser.add_argument('--goals_size', type=int, default=2)
    parser.add_argument('--pretrained_resnet', type=int, default=1)

    parser.add_argument('--slam_batch_size', type=int, default=72)
    parser.add_argument('--slam_iterations', type=int, default=10)
    parser.add_argument('--slam_memory_size', type=int, default=500000)
    parser.add_argument('--proj_loss_coeff', type=float, default=1.0)
    parser.add_argument('--pose_loss_coeff', type=float, default=10000.0)
    parser.add_argument('--exp_loss_coeff', type=float, default=1.0)
    parser.add_argument('--global_downscaling', type=float, default=2)
    parser.add_argument('--map_pred_threshold', type=float, default=0.5)

    parser.add_argument('--vision_range', type=int, default=64)
    parser.add_argument('--obstacle_boundary', type=int, default=5)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=2)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--obs_threshold', type=float, default=1)
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    parser.add_argument('--noise_level', type=float, default=1.0)
    
    parser.add_argument('--ft_num_local_steps', type=int, default = 15)
    parser.add_argument('--ft_clear_radius',type=float, default = 50)
    parser.add_argument('--ft_cluster_radius', type=float, default = 5.0)
    parser.add_argument('--ft_use_random', default = False, action='store_true')
    parser.add_argument('--rrt_expand_dis', type = float, default = 2.0)
    parser.add_argument('--rrt_goal_sample_rate', type = float, default = 20)
    parser.add_argument('--rrt_max_iter', type = int, default = 300)
    parser.add_argument('--rrt_connect_circle_dist', type = float, default = 10.0)
    parser.add_argument('--utility_radius', type = int, default = 30)

    parser.add_argument('--grid_size', default=8, type=int)
    
    # grid goal global policy
    parser.add_argument('--grid_goal', default=False, action='store_true')
    parser.add_argument('--grid_goal_simpler', default=False, action='store_true')
    parser.add_argument('--use_grid_simple', default=False, action='store_true')
    parser.add_argument('--discrete_goal', action='store_true',
                        default=False, help="by default True, use_local_map")
    parser.add_argument('--multidiscrete_goal', action='store_true',
                        default=False, help="by default True, use_local_map")
    parser.add_argument('--use_goal_penalty', action='store_true',
                        default=False, help="by default True, use_local_map")
    parser.add_argument('--action_mask', action='store_true',
                        default=False, help="by default True, action_mask")
    parser.add_argument('--use_scene_norm', action='store_true',
                        default=False, help="by default True, use_scene_norm")
    parser.add_argument('--use_enhanced_id', action='store_true',
                        default=False, help="by default True, use_scene_norm")
    parser.add_argument('--use_one_cnn_model', action='store_true',
                        default=False, help="by default True, use_scene_norm")
    parser.add_argument('--use_share_cnn_model', action='store_true',
                        default=False, help="by default True, use shared cnn parameters")
    
    # direction goal
    parser.add_argument('--direction_goal', default=False, action='store_true')
    parser.add_argument('--direction_mode', default='discrete', choices = ['discrete', 'continuous'])
    parser.add_argument('--direction_k', default=8, type=int)

    # graph
    parser.add_argument('--build_graph', default=False, action='store_true')
    parser.add_argument('--feature_dim', default=512, type=int)
    parser.add_argument('--graph_memory_size', default=100, type=int)
    parser.add_argument('--paro_frame_width', default=252, type=int)
    parser.add_argument('--paro_frame_height', default=64, type=int)
    parser.add_argument('--paro_hfov', type=int, default=30,
                        help="horizontal field of view in degrees")
    parser.add_argument('--time_dim', default=8, type=int)
    parser.add_argument('--num_recurrent_layers', default=2, type=int)
    parser.add_argument('--graph_th', type = float, default = 0.75)
    parser.add_argument('--paro_position_y', type = float, default = 1.25)
    parser.add_argument('--direct_action', default=False, action='store_true')
    parser.add_argument('--use_idx_embedding', default=False, action='store_true')
    parser.add_argument('--add_ghost', default=False, action='store_true')
    parser.add_argument('--use_direct_action', default=False, action='store_true')
    parser.add_argument('--several_direct_actions', default=False, action='store_true')
    parser.add_argument('--use_num_embedding', default=False, action='store_true')
    parser.add_argument('--use_merge', default=False, action='store_true')
    parser.add_argument('--use_global_goal', default=False, action='store_true')
    parser.add_argument('--use_action_embedding', default=False, action='store_true')
    parser.add_argument('--use_each_node', default=False, action='store_true')
    parser.add_argument('--use_goal_embedding', default=False, action='store_true')
    parser.add_argument('--use_restrict_graph', default=False, action='store_true')
    parser.add_argument('--use_prediction', default=False, action='store_true')
    parser.add_argument('--use_frontier', default=False, action='store_true')
    parser.add_argument('--cut_ghost', default=False, action='store_true')
    parser.add_argument('--learn_to_build_graph', default=False, action='store_true')
    parser.add_argument('--use_other_agent', default=False, action='store_true')
    parser.add_argument('--use_edge_info', default=None, choices=['learned', 'hard'])
    parser.add_argument('--use_tail_agent_info', default=False, action='store_true')
    parser.add_argument('--same_node_penalty', default=False, action='store_true')
    parser.add_argument('--new_node_reward', default=False, action='store_true')
    parser.add_argument('--use_agent_node', default=False, action='store_true')
    parser.add_argument('--use_distance', default=False, action='store_true')
    parser.add_argument('--overlap_node_penalty', default=False, action='store_true')
    parser.add_argument('--use_mgnn', default=False, action='store_true')
    parser.add_argument('--use_map_critic', default=False, action='store_true')
    parser.add_argument('--use_mtsp', default=False, action='store_true')
    parser.add_argument('--dis_gap', default=2, type=int)
    parser.add_argument('--use_frontier_nodes', default=False, action='store_true')
    parser.add_argument('--max_frontier', default=50, type=int)
    parser.add_argument('--use_several_nodes', default=False, action='store_true')
    parser.add_argument('--use_all_ghost_add', default=False, action='store_true')
    parser.add_argument('--use_ghost_goal_penalty', default=False, action='store_true')
    parser.add_argument('--use_batch_train', default=False, action='store_true')
    parser.add_argument('--use_local_frontier', default=False, action='store_true')
    parser.add_argument('--use_submap', default=False, action='store_true')
    parser.add_argument('--sub_size', default=64, type=int)
    parser.add_argument('--ghost_node_size', default=12, type=int)
    parser.add_argument('--use_only_agent_graph', default=False, action='store_true')
    parser.add_argument('--use_history_cross', default=False, action='store_true')
    parser.add_argument('--use_async', default=False, action='store_true')
    parser.add_argument('--use_gt_map', default=False, action='store_true')
    parser.add_argument('--use_subratio', default=False, action='store_true')
    parser.add_argument('--use_double_matching', default=False, action='store_true')
    parser.add_argument('--use_mtsp2', default=False, action='store_true')
    parser.add_argument('--matching_type', type=str)
    parser.add_argument('--use_here_rotation', default=False, action='store_true')
    parser.add_argument('--use_clear_zone', default=False, action='store_true')
    parser.add_argument('--use_noisy_render', default=False, action='store_true')
    parser.add_argument('--use_ft_frontiers', default=False, action='store_true')
    
    # image retrieval
    parser.add_argument('--encoder_url', type=str, default=None)
    parser.add_argument("--img_resol", default=1024, type=int)
    parser.add_argument('--whitening', default=None, choices=['retrieval-SfM-30k', 'retrieval-SfM-120k'])
    parser.add_argument('--use_retrieval', default=False, action='store_true')
    parser.add_argument('--rotate_obs', default=False, action='store_true')
    parser.add_argument('--update_nodes', default=False, action='store_true')
    parser.add_argument('--use_sparse_graph', default=True, action='store_false')
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.episode_length = all_args.max_episode_length // all_args.num_local_steps
    print("global episode length is {}. \n".format(all_args.episode_length))

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), (
            "check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy ==
                False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[
                                 1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        if all_args.use_async:
            from onpolicy.runner.shared.async_graph_habitat_runner import GraphHabitatRunner as Runner
        else:
            from onpolicy.runner.shared.graph_habitat_runner import GraphHabitatRunner as Runner
    else:
        from onpolicy.runner.separated.habitat_runner import HabitatRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if not all_args.use_render:
        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(
                str(runner.log_dir + '/summary.json'))
            runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])
