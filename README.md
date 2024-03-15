# ON-POLICY

## support algorithms

| Algorithms | recurrent-verison | mlp-version | cnn-version | share-base version | independent version |
| :--------: | :---------------: | :---------: | :---------: |:---------------: |:---------------: |
| MAPPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark:        |:heavy_check_mark:        |
| MAPPG |        :heavy_check_mark:           |       :heavy_check_mark:      |     :heavy_check_mark:        |:heavy_check_mark:        |:heavy_check_mark:        |
| MATRPO[^1] |        :heavy_check_mark:           |       :heavy_check_mark:      |     :heavy_check_mark:        |:heavy_check_mark:        |:heavy_check_mark:        |

[^1]: see trpo branch


## support environments:
**Pay Attention:** we sometimes hack the environment code to fit our task and setting. 
- [StarCraftII](https://github.com/oxwhirl/smac)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
- [MPE](https://github.com/openai/multiagent-particle-envs)
- [Hide-and-Seek](https://github.com/openai/multi-agent-emergence-environments)
- [social dilemmas](https://github.com/eugenevinitsky/sequential_social_dilemma_games)
- [agar.io](https://github.com/staghuntrpg/agar)
- [SMARTS](https://github.com/huawei-noah/SMARTS)
- [HighWay](https://github.com/eleurent/highway-env)
- [Habitat](https://github.com/facebookresearch/habitat-sim)
- [Gibson](https://github.com/StanfordVL/GibsonEnv)
- [Gibson2](https://github.com/StanfordVL/iGibson)
- [Mini-GridWorld](https://github.com/maximecb/gym-minigrid)
- [StagHunt](https://github.com/staghuntrpg/RPG/tree/main/GridWorld)
- [MultiVehicleEnv](https://github.com/efc-robot/MultiVehicleEnv.git)

## TODOs:
- [ ] multi-agent FLOW

## 1. Install

### 1.1 instructions

   test on CUDA == 10.1   

``` Bash
   conda create -n marl
   conda activate marl
   pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   cd onpolicy
   pip install -e . 
```

### 1.2 hyperparameters

* config.py: contains all hyper-parameters

* default: use GPU, chunk-version recurrent policy and shared policy

* other important hyperparameters:
  - use_centralized_V: Centralized training (MA) or Centralized training (I)
  - use_single_network: share base or not
  - use_recurrent_policy: rnn or mlp
  - use_eval: turn on evaluation while training, if True, u need to set "n_eval_rollout_threads"
  - wandb_name: For example, if your wandb link is https://wandb.ai/mapping, then you need to change wandb_name to "mapping". 
  - user_name: only control the program name shown in "nvidia-smi".

## 2. StarCraftII

### 2.1 Install StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc
```

*  download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

*  If you want stable id, you can copy the `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

### 2.2 Train StarCraftII

* train_smac.py: all train code

  + Here is an example:


``` Bash
  conda activate marl
  cd scripts
  chmod +x train_smac.sh
  ./train_smac.sh
```

  + local results are stored in fold `scripts/results`, if you want to see training curves, login wandb first, see guide [here](https://docs.wandb.com/). Sometimes GPU memory may be leaked, you need to clear it manually.   

``` Bash
   ./clean_gpu.sh
```

### 2.3 Tips

   Sometimes StarCraftII exits abnormally, and you need to kill the program manually.

``` Bash
   ./clean_smac.sh
   ./clean_zombie.sh
```

if you want to run MADDPG/MATD3/MASAC algorithms, welcome to use this repository [offpolicy](https://github.com/marlbenchmark/offpolicy)
## 3. Hanabi

  ### 3.1 Hanabi

   The environment code is reproduced from the hanabi open-source environment, but did some minor changes to fit the algorithms. Hanabi is a game for **2-5** players, best described as a type of cooperative solitaire.

### 3.2 Install Hanabi 

``` Bash
   pip install cffi
   cd envs/hanabi
   mkdir build & cd build
   cmake ..
   make -j
```

### 3.3 Train Hanabi

   After 3.2, we will see a libpyhanabi.so file in the hanabi subfold, then we can train hanabi using the following code.

``` Bash
   conda activate onpolicy
   cd scripts
   chmod +x train_hanabi_forward.sh
   ./train_hanabi_forward.sh
```
we also have a backward version training script, which uses a different way to calculate reward of one turn.

``` Bash
   conda activate onpolicy
   cd scripts
   chmod +x train_hanabi_backward.sh
   ./train_hanabi_backward.sh
```

## 4. MPE

### 4.1 Install MPE

``` Bash
   # install this package first
   pip install seaborn
```

3 Cooperative scenarios in MPE:

* simple_spread: set num_agents=3
* simple_speaker_listener: set num_agents=2, and use --share_policy
* simple_reference: set num_agents=2

### 4.2 Train MPE   

``` Bash
   conda activate marl
   cd scripts
   chmod +x train_mpe.sh
   ./train_mpe.sh
```

## 5. Hide-And-Seek

we support multi-agent boxlocking and blueprint_construction tasks in the hide-and-seek domain.

### 5.1 Install Hide-and-Seek

#### 5.1.1 Install MuJoCo

1. Obtain a 30-day free trial on the [MuJoCo website](https://www.roboti.us/license.html) or free license if you are a student. 

2. Download the MuJoCo version 2.0 binaries for [Linux](https://www.roboti.us/download/mujoco200_linux.zip).

3. Unzip the downloaded `mujoco200_linux.zip` directory into `~/.mujoco/mujoco200`, and place your license key at `~/.mujoco/mjkey.txt`.

4. Add this to your `.bashrc` and source your `.bashrc`.


``` 
   export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
```

#### 5.1.2 Intsall mujoco-py and mujoco-worldgen

1. You can install mujoco-py by running `pip install mujoco-py==2.0.2.13`. If you encounter some bugs, refer this official [repo](https://github.com/openai/mujoco-py) for help.

   ```
   sudo apt-get install libgl1-mesa-dev libosmesa6-dev
   ```

2. To install mujoco-worldgen, follow these steps:


``` Bash
    # install mujuco_worldgen
    cd envs/hns/mujoco-worldgen/
    pip install -e .
    pip install xmltodict
    # if encounter enum error, excute uninstall
    pip uninstall enum34
```

### 5.2 Train Tasks

``` Bash
   conda activate marl
   # boxlocking task, if u want to train simplified task, need to change hyper-parameters in box_locking.py first.
   cd scripts
   chmod +x train_boxlocking.sh
   ./train_boxlocking.sh
   # blueprint_construction task
   chmod +x train_bpc.sh
   ./train_bpc.sh
   # hide and seek task
   chmod +x train_hns.sh
   ./train_hns.sh
```

## 6. Flow

### 6.1 install sumo

```Bash
cd envs/decentralized_bottlenecks/scripts

# choose the bash scripts according to your platform
./setup_sumo_ubuntu1604.sh 

# default write the PATH to ~/.bashrc, if you are using zsh, copy the PATH to ~/.zshrc
source ~/.zshrc

# check whether the sumo is installed correctly
which sumo
sumo --version
sumo-gui
```

### 6.2 install flow

```Bash
pip install lxml imutils gym-0.10.5

# check whether your flow is installed correctly
python examples/sumo/sugiyama.py
```

## 7. SMARTS


1. git clone sumo, pay attention to use sumo version < 1.8

2. `cmake ../.. & make -j` sumo and `make install` sumo, u can use `sumo` in the terminal, then u can see the version of sumo.

   [^sumo]: if u encounter TIFF error, conduct: `conda remove libtiff==4.1.0`, actually we need to use `conda install libtiff==4.0.9`.
   
3. git clone smarts and `pip install -e .`[please remove some unneeded packages in requirement.txt]

4. `scl scenario build --clean ./loop` loop is ur own scenerio.

5. all is ready , enjoy `./train_smarts.sh`

## 8. HighWay

1. training script: `./train_highway.sh`
1. rendering script `./render_highway.sh`

## 9.human

1. training script: `./train_human.sh`
2. rendering script: `./render_human.sh`
   When you type `--use_human_command` in the script `render_human.sh`, You can tell the predator the location of the prey by inputting command. When you press 0, it means the prey is on the top right of the predator. Similarly, 1 means lower right, 2 means top left and 3 means lower left. In addition, the brighter the predator, the bigger the id.

## 10. Gibson2

```
cd onpolicy
# git submodule init 
# git submodule update
git submodule update --init --recursive
cd onpolicy/envs/iGibson
git submodule update --init --recursive

# if u want to use the original repo, use the following command instead of the above one.
# git clone https://github.com/StanfordVL/iGibson --recursive

pip install -e .
```

If you failed to clone pybind11, use the following command.

```
cd iGibson
git submodule update
```

If u have installed IGibson successfully, then u can download dataset.
```
cd onpolicy/envs/iGibson/gibson2
mkdir data
cd data
wget https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz
wget https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz
tar -zxvf ig_dataset.tar.gz
tar -zxvf assets_igibson.tar.gz
```

Note: we support using a custom pybullet version to speed up the physics in iGibson, if you want to have the speed up, you would need to do the following steps after installation:

```
pip uninstall pybullet
pip install https://github.com/StanfordVL/bullet3/archive/master.zip
```

If you have updated submodules, use the following command to synchronize the updates into onpolicy repository.
```
# single update
git submodule foreach git checkout master
# batch update
git submodule foreach git submodule update
```

## 11. habitat


``` 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple magnum scikit-image==0.17.2 lmdb scikit-learn==0.24.1 scikit-fmm yacs imageio-ffmpeg numpy-quaternion numba tqdm gitpython attrs==19.1.0 tensorboard
```
tips: IF u encounter errors, try to use `--ignore-installed`.
```
cd onpolicy
git submodule update --init --recursive
cd habitat/habitat-sim
./build.sh --headless # make sure you use sh file!!!!!!
cd habitat/habitat-lab
pip install -e .
# if you failed to install habitat-api, you can use `build.sh --headless` instead.
```

Remember to add PYTHONPATH in your ~/.bashrc file:
```
export PYTHONPATH=$PYTHONPATH:/home/yuchao/project/onpolicy/onpolicy/envs/habitat/habitat-sim/
```

```
cd /home/yuchao/project/onpolicy/onpolicy/envs/habitat
mkdir data/datasets
cd data/datasets
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip
unzip pointnav_gibson_v1.zip
ln -s /mnt/disk2/nav/habitat_data/scene_datasets
```
## 12.Human-Robot

## 13.Mini-GridWorld

## 14.StagHunt

## 15. MultiVehicleEnv(MVE)
This repo is supported as a submodule of on-policy repo, use the following command to clone MVE environment.

```
cd onpolicy
git submodule update --init --recursive
```

```
cd onpolicy/onpolicy/envs/MultiVehicleEnv
cd src
pip install -e .
```


## 16. Docs：

```
pip install sphinx sphinxcontrib-apidoc sphinx_rtd_theme recommonmark

sphinx-quickstart
make html
```

## 16. submodules

here we give an example on how to add your repo as a submodule of on-policy repo
```
git submodule add https://github.com/zoeyuchao/habitat-api.git

# add source for syncing
git remote add dist_source https://github.com/facebookresearch/habitat-lab.git
git remote -v

```
If u want to sync the official updates, you can use the following command.
```
git pull dist_source master
# after you fix merging conflict, then you can merge into master branch 
git push origin master
```
When you update your submodule, you need to update the main repo, using the following command.
```
git submodule foreach git submodule update
```
