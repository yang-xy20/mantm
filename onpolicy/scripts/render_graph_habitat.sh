#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="new_mgnn_new_critic(*15)_new_dis(3)_step15_render"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6,7 python eval/eval_graph_habitat.py --scenario_name ${scenario} --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --num_agents ${num_agents} --split "train" --use_same_scene --scene_id 20 --eval_episodes 100 --use_eval \
    --ifi 0.01 --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 300 --num_local_steps 15 \
    --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 --use_maxpool2d \
    --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1' \
    --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --log_interval 1 \
    --load_local "../envs/habitat/model/pretrained_models/local_best.pt" \
    --model_dir "./results/Habitat/mappo/new_mgnn_new_critic(*15)_new_dis(3)_step15/wandb/run-20220521_194614-288qnui0/files" --use_centralized_V \
    --use_delta_reward --use_merge_partial_reward --use_time_penalty \
    --use_overlap_penalty --use_complete_reward --wandb_name mapping --user_name yang-xy20 \
    --feature_dim 512 --hidden_size 256 --use_different_start_pos --build_graph \
    --use_fixed_start_pos --use_recurrent_policy --use_merge --add_ghost \
    --use_mgnn --use_global_goal --use_local_single_map --cut_ghost \
    --graph_memory_size 100 --use_restrict_graph --dis_gap 3 --use_map_critic \
    --use_max --use_render --save_gifs --use_wandb --learn_to_build_graph \
    --use_all_ghost_add --ghost_node_size 24
    echo "training is done!" 
done

