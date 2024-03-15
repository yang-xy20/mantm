#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import GuardSubprocVecEnv, DummyVecEnv, ChooseGuardSubprocVecEnv, ChooseSimpleDummyVecEnv
from onpolicy.envs.smarts.SMARTS_Env import SMARTSEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SMARTS":
                env=SMARTSEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return GuardSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SMARTS":
                env=SMARTSEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseSimpleDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseGuardSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # smarts parameters, need to include these parameters in the bash file.
    #parser.add_argument("--scenario_path", type=str, default='../envs/smarts/SMARTS/scenarios/')
    parser.add_argument("--scenario_path", type=str, default='/home/jiangz/SMARTS/scenarios/')
    parser.add_argument('--scenario_name', type=str, default='straight', help="Which scenario to run")
    
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument('--horizon', type=int, default=100, help="max step of task")
    parser.add_argument("--use_proximity", action="store_true", default=False)
    parser.add_argument("--use_discrete", action="store_false", default=True)

    parser.add_argument("--rews_mode", type=str, default="vanilla", help="used to specify env's rew")
    parser.add_argument('--neighbor_num', type=int, default=3, help="number of neighbor you can see in the env")

    # sumo parameters, u'd better to use the default value.
    parser.add_argument("--headless", help="true|false envision disabled", action="store_true", default=False)
    parser.add_argument("--visdom", help="true|false visdom integration", action="store_true", default=False)
    parser.add_argument("--shuffle_scenarios", action="store_false", default=True)

    parser.add_argument("--sumo_auto_start", help="true|false sumo will start automatically", action="store_false", default=True)
    parser.add_argument("--sumo_headless", help="true|false for SUMO visualization disabled [sumo-gui|sumo]", action="store_false", default=True)
    parser.add_argument("--sumo_port", help="used to specify a specific sumo port.", type=int, default=None)
    parser.add_argument("--num_external_sumo_clients", help="the number of SUMO clients beyond SMARTS", type=int, default=0)
    parser.add_argument("--timestep_sec", help="the step length for all components of the simulation", type=float, default=0.1)

    parser.add_argument("--auth_key", type=str, default=None, help="Authentication key of type string for communication with Zoo Workers")
    parser.add_argument("--zoo_workers", type=str, default=None, help="List of (ip, port) tuples of Zoo Workers, used to instantiate remote social agents")
    
    parser.add_argument("--envision_record_data_replay_path", type=str, default=None, help="used to specify envision's data replay output directory")
    parser.add_argument("--envision_endpoint", type=str, default=None, help="used to specify envision's uri")
    parser.add_argument("--endless_traffic", help="Run the simulation in endless mode.", action="store_false", default=True)
 
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
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
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
        from onpolicy.runner.shared.smarts_runner import SMARTSRunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

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
