#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import gibson2
import torch

from onpolicy.config import get_config

from onpolicy.envs.iGibson.iGibson_Env import iGibsonEnv
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import gibson2

def get_scene_files():
    scenes=[]
    content_dir = os.path.join(gibson2.ig_dataset_path,'scenes')
    scene_dataset_ext = '_int'
    if not os.path.exists(content_dir):
        return scenes
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename
            scenes.append(scene)
    scenes.sort()
    return scenes

def make_train_env(all_args, scene_files):
    def get_env_fn(rank):
        scene_id = scene_files[rank % len(scene_files)] if scene_files is not None else None
        def init_env():
            if all_args.env_name == "iGibson":
                env = iGibsonEnv(all_args, scene_id)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, scene_files):
    def get_env_fn(rank):
        scene_id = scene_files[rank % len(scene_files)] if scene_files is not None else None
        def init_env():
            if all_args.env_name == "iGibson":
                env = iGibsonEnv(all_args, scene_id)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='locobot_point_nav', help="Which scenario to run on")
    parser.add_argument('--mode', '-m', choices=['headless', 'gui', 'iggui'], default='headless',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument('--render_gpu_id', type=int,
                        default=0, help="which gpu is used to render")
    #scene parameters
    parser.add_argument("--use_different_scenes", action='store_true',
                        default=False, help="identify scene_id.")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
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
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
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
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    scene_files = get_scene_files() if all_args.use_different_scenes else None
    envs = make_train_env(all_args, scene_files)
    eval_envs = make_eval_env(all_args, scene_files) if all_args.use_eval else None
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
        from onpolicy.runner.shared.igibson_runner import iGibsonRunner as Runner
    else:
        from onpolicy.runner.separated.igibson_runner import iGibsonRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
