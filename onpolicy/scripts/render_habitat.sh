#!/bin/sh
env="Habitat"
scenario="pointnav_gibson"
num_agents=2
algo="mappo"
exp="hm3d_10"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1 python eval/eval_habitat.py --scenario_name ${scenario} --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --num_agents ${num_agents} --split "train" \
    --use_same_scene --scene_id 10 --eval_episodes 100 --use_eval --ifi 0.01 --seed 1 --n_training_threads 1 \
    --n_rollout_threads 1 --num_mini_batch 5 --max_episode_length 1800 --num_local_steps 15 \
    --num_env_steps 20000000 --ppo_epoch 4 --gain 0.01 --lr 2.5e-5 --critic_lr 2.5e-5 \
    --use_maxpool2d --cnn_layers_params '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1' \
    --hidden_size 256 --log_interval 1 --use_recurrent_policy  \
    --load_slam "../envs/habitat/model/pretrained_models/slam_best.pt" \
    --load_local "../envs/habitat/model/pretrained_models/local_best.pt" \
    --model_dir "./results/Habitat/mappo/distill" \
    --use_centralized_V --use_delta_reward --use_merge_partial_reward --use_time_penalty \
    --use_overlap_penalty --use_complete_reward --use_single --wandb_name "mapping" --use_max \
    --use_new_trace --use_weight_trace --use_goal --use_own --grid_goal --use_grid_simple --grid_pos \
    --grid_last_goal --cnn_use_transformer --use_share_cnn_model --agent_invariant --invariant_type alter_attn \
    --use_different_start_pos --use_pos_embedding --multi_layer_cross_attn \
    --use_self_attn --use_intra_attn --add_grid_pos --use_id_embedding --dataset 'hm3d' \
    --use_gt_map --global_downscaling 3 --map_size_cm 3600 --use_fixed_start_pos
    echo "training is done!" 
done

#--global_downscaling 3 --map_size_cm 3600 