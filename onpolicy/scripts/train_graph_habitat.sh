#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=3
algo="mappo"
exp="only20_mgnn_gt_graph_new_critic(*15)_new_dis(3)_step15_goal_penalty"

seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1,2 python train/train_graph_habitat.py --scenario_name ${scenario} \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} \
    --split "train" --seed 1 --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 5 \
    --num_local_steps 15 --max_episode_length 300 --num_env_steps 3000000 --ppo_epoch 4 --gain 0.01 \
    --lr 5e-4 --critic_lr 2.5e-5 --use_maxpool2d \
    --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1' \
    --log_interval 1 --load_local "../envs/habitat/model/pretrained_models/local_best.pt" --save_interval 10 \
    --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" --use_centralized_V \
    --eval_episodes 1 --use_same_scene --scene_id 20 \
    --use_delta_reward --use_merge_partial_reward --use_time_penalty \
    --use_complete_reward --wandb_name mapping --user_name yang-xy20 \
    --feature_dim 512 --hidden_size 256 --use_different_start_pos --build_graph \
    --use_recurrent_policy --use_merge --use_mgnn --use_global_goal --use_local_single_map \
    --graph_memory_size 100 --use_restrict_graph --dis_gap 3 --ghost_node_size 24 \
    --use_map_critic --use_max --add_ghost --learn_to_build_graph --use_ghost_goal_penalty \
    --cut_ghost --use_all_ghost_add --entropy_coef 0.0001 --value_loss_coef 3 \
    --use_double_matching --matching_type multi --global_downscaling 3 --map_size_cm 3600 --use_async --use_wandb

    echo "training is done!" 
done