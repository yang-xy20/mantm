import random
import torch.nn as nn
import torch
import numpy as np
import os
from icecream import ic
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
import math
import numpy as np
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.envs.habitat.model.PCL.resnet_pcl import resnet18
import onpolicy
import wandb

class Args():
    def __init__(self):
        self.map_nums = 24
        self.attn_heads = 8
        self.epi_nums = 10
        self.feature_size = 512
        self.zone_nums = 10
        self.num_steps = 602
        self.embedding_size = 128
        self.lr = 1e-4
        self.wd = 0.1
        self.train_maps = list(range(24))
        self.temperature = 0.1
        self.model_path = None
        self.k_test = 2
        self.gpu_id = 2
        self.epoch = 10
        self.k_fold = 4
        self.batch_size = 5
        self.eps = 1e-5
        self.img_shape = [4, 64, 252]
        self.path = '/home/tsing69/hpc/usr-9yS2URci/base/samples'

def trans_angle(img, angle):
    ratio = angle/360
    idx = int(ratio*img.shape[1])
    new_img = torch.cat([img[:,idx:,:],img[:,0:idx,:]], dim=1)
    return new_img

def read_data(args):
    path = args.path
    map_nums = args.map_nums
    epi_nums = args.epi_nums
    img_data = np.zeros([map_nums, epi_nums, args.num_steps, 4, 64, 252])
    pos_data = np.zeros([map_nums, epi_nums, args.num_steps, 3])
    step_data = np.zeros([map_nums, epi_nums, args.num_steps])
    levels = os.listdir(path)
    episode_idx = 0
    map_idx = 0
    for level in levels:
        tmaps = os.listdir(path+'/'+str(level))
        for tmap in tmaps:
            episodes = os.listdir(path+'/'+str(level)+'/'+str(tmap))[:epi_nums]
            for episode in episodes:
                file_dir = path+'/'+str(level)+'/'+str(tmap)+'/'+str(episode)+'/'
                img_data[map_idx, episode_idx,:,:3,:,:] = np.load(file_dir+str('img.npy')).transpose((0,3,1,2))
                pos_data[map_idx, episode_idx] = np.load(file_dir+str('pos.npy'))
                img_data[map_idx, episode_idx,:,3:,:,:] = np.load(file_dir+str('depth.npy')).transpose((0,3,1,2))
                step_data[map_idx, episode_idx] = np.load(file_dir+str('step.npy'))
                episode_idx += 1
            map_idx += 1
            episode_idx = 0
    ic('data loaded!')
    img_data[:,:,:,:3,:,:] = img_data[:,:,:,:3,:,:]/255
    return img_data, pos_data, step_data

class pre_process():
    def __init__(self, args):
        self.args = args
        self.processor = self.load_visual_encoder(self.args.img_shape, self.args.feature_size)
        self.final_embed = np.zeros([self.args.map_nums, self.args.epi_nums, self.args.num_steps, self.args.feature_size])

    def load_visual_encoder(self, input_shape, feature_dim):
        visual_encoder = resnet18(num_classes=feature_dim)
        dim_mlp = visual_encoder.fc.weight.shape[1]
        visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
        ckpt_pth = onpolicy.__path__[0]+ "/envs/habitat/model/PCL/PCL_encoder.pth"
        ckpt = torch.load(ckpt_pth, map_location='cpu')
        visual_encoder.load_state_dict(ckpt)
        visual_encoder.eval()
        return visual_encoder

    def process(self):
        self.args = Args()
        img_data, pos_data, _ = read_data(self.args)
        img = torch.tensor(img_data,dtype=torch.float)
        for mapid in range(self.args.map_nums):
            for epi_id in range(self.args.epi_nums):    
                self.final_embed[mapid, epi_id] = nn.functional.normalize(self.processor(img[mapid, epi_id])).detach()
        np.save('data.npy', self.final_embed)
        np.save('pos.npy', pos_data)
        return self.final_embed, pos_data

    def clust(self, map_id=None, epi_id=None, pos=None):
        args =self.args
        x = []
        y = []
        label = []
        average_cluster = []
        self.final_embed = np.load('data.npy')
        show_flag = True if map_id is not None and epi_id is not None else False
        cluster = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.2, compute_full_tree=True)
        #cluster = AgglomerativeClustering(n_clusters=args.zone_nums)
        #self.zones = np.zeros([self.embed_img.shape[0], self.embed_img.shape[1], args.num_steps]) 
        zone_data = {}#which map, which episode, which step(which zone)
        for mapid in range(args.map_nums):
            for epi in range(args.epi_nums):
                cluster.fit(self.final_embed[mapid, epi])
                average_cluster.append(np.amax(list(cluster.labels_))+1)
                #self.zones[mapid, epi] = cluster.labels_                    
                for idx in range(len(cluster.labels_)):
                    if show_flag == True:
                        if mapid == map_id and epi == epi_id:
                            if cluster.labels_[idx] is not None:
                                x.append(pos[mapid, epi_id, idx][0])
                                y.append(pos[mapid, epi_id, idx][1])
                                label.append(cluster.labels_[idx])
                    if str(mapid)+'-'+str(epi)+'-'+str(cluster.labels_[idx]) not in zone_data.keys():
                        zone_data[str(mapid)+'-'+str(epi)+'-'+str(cluster.labels_[idx])] = []
                    zone_data[str(mapid)+'-'+str(epi)+'-'+str(cluster.labels_[idx])].append([mapid, epi, idx, cluster.labels_[idx]])
        if show_flag is True:
            df = pd.DataFrame({'x':x,'y':y,'label':label})
            sns.scatterplot(x = "x", y = "y",data=df, hue='label')
            sns.set(style='whitegrid')
            plt.show()
        ic('clust done!')
        ic(np.mean(average_cluster))
        return zone_data

class myData(data.Dataset):
    def __init__(self, zone):
        self.zone = zone

    def __getitem__(self, index):
        key = list(self.zone.keys())[index]
        return key, self.zone[key]

    def __len__(self):
        return len(list(self.zone.keys()))

class myLoader(object):
    def __init__(self, data, batch_size, shuffle=False):
        self.dataset = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = list(range(len(data)))
        if self.shuffle:
            random.shuffle(self.sampler)
        self.index = 0

    def __next__(self):
        if self.index < len(self.sampler):
            if self.index + self.batch_size <= len(self.sampler):
                indice = self.sampler[self.index:self.index+self.batch_size]
            else:
                indice = self.sampler[self.index:]
            batch_data = {}
            for i in range(len(indice)):
                key,value = self.dataset[indice[i]]
                batch_data[key] = value
            self.index += self.batch_size
        else:
            self.__init__(self.dataset, self.batch_size, self.shuffle)
            raise StopIteration
        return batch_data

    def __iter__(self):
        return self

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FeedForward(nn.Module):
    def __init__(self, d_model, output_shape=128, d_ff=512, dropout=0.0, use_orthogonal=True, activation_id=1):
        super(FeedForward, self).__init__()
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.linear_1 = nn.Sequential(
            init_(nn.Linear(d_model, d_ff)), active_func, nn.LayerNorm(d_ff))

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, output_shape))

    def forward(self, x):
        x = self.dropout(self.linear_1(x))
        x = self.linear_2(x)
        return x


def ScaledDotProductAttention(q, k, v, d_k, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MHAttentionBlock(nn.Module):
    def __init__(self, heads, hidden_size, dropout, use_orthogonal, activation_id):
        super(MHAttentionBlock, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.hidden_size = hidden_size
        self.d_k = hidden_size // heads
        self.h = heads

        self.q_linear = init_(nn.Linear(hidden_size, hidden_size))
        self.v_linear = init_(nn.Linear(hidden_size, hidden_size))
        self.k_linear = init_(nn.Linear(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.out = init_(nn.Linear(hidden_size, hidden_size))
        self.mlp = FeedForward(self.hidden_size,output_shape=self.hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x, y):
        # perform linear operation and split into h heads
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, dim=0)
        xbs = x.size(0)
        ybs = y.size(0)
        k = self.k_linear(y).view(ybs, self.h, self.d_k)
        q = self.q_linear(x).view(xbs, self.h, self.d_k)
        v = self.v_linear(y).view(ybs, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * hidden_size

        k = k.transpose(1, 0)
        q = q.transpose(1, 0)
        v = v.transpose(1, 0)
        # calculate attention
        scores = ScaledDotProductAttention(
            q, k, v, self.d_k, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 0).contiguous()\
            .view(xbs, -1, self.hidden_size)
        if len(x.shape) != len(concat.shape):
            x = torch.unsqueeze(x,dim=1)
        output = self.dropout2(self.out(concat)) + x
        output = self.ln(output)
        output = self.ln2(self.dropout3(self.mlp(output)) + output)
        return torch.squeeze(output)

class Embedding(nn.Module):
    def __init__(self, img_shape, output_shape, use_orthogonal, activation_id):
        super(Embedding, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.pos_embedding = nn.Sequential(init_(
                nn.Linear(3, img_shape)), active_func)
        self.output_proj = nn.Sequential(init_(nn.Linear(2*img_shape, output_shape)), active_func, nn.LayerNorm(output_shape))
        self.norm = nn.LayerNorm(2*img_shape)

    def forward(self, img, pos):
        pos_embed = self.pos_embedding(pos)
        output = torch.cat([img, pos_embed], dim=-1)
        output = self.norm(output)
        output = self.output_proj(output)
        return output

class Encoder(nn.Module):
    def __init__(self, img_shape, d_model, heads, dropout, use_orthogonal, activation_id):
        super(Encoder, self).__init__()
        self.embedding = Embedding(img_shape, d_model, use_orthogonal, activation_id)
        self.attn = MHAttentionBlock(heads, d_model, dropout, use_orthogonal, activation_id)


    def forward(self, x, pos):
        hidden = self.embedding(x, pos)
        output = self.attn(hidden, hidden)
        return output


class Decoder(nn.Module):
    def __init__(self, img_shape, d_model, heads, dropout, use_orthogonal, activation_id):
        super(Decoder, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.embedding = nn.Sequential(init_(nn.Linear(3, d_model)), active_func, nn.LayerNorm(d_model))
        self.attn = MHAttentionBlock(heads, d_model, dropout, use_orthogonal,activation_id)
        self.mlp = FeedForward(d_model, img_shape)

    def forward(self, pos, env):
        q = self.embedding(pos)
        output = self.attn(q, env)
        output = self.mlp(output)
        return output

class Predictor(nn.Module):
    def __init__(self, heads, img_shape, d_model, dropout=0.1, use_orthogonal=True, activation_id=1):
        super(Predictor, self).__init__()
        self.encoder = Encoder(img_shape, d_model, heads, dropout, use_orthogonal, activation_id)
        self.decoder = Decoder(img_shape, d_model, heads, dropout, use_orthogonal, activation_id)
    
    def forward(self, img, pos, pre_pos):
        embedded = self.encoder(img, pos)
        out_feature = self.decoder(pre_pos, embedded)
        return out_feature

class sim(nn.Module):
    def __init__(self, dim, eps):
        super(sim, self).__init__()
        self.calcu = nn.CosineSimilarity(dim, eps)
    
    def forward(self, x, y):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x,dim=0)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y,dim=0)
        return self.calcu(x, y)

class trainer():
    def __init__(self, args, img, pos, zone_data):
        super(trainer, self).__init__()
        # img = [map*episode*step*feature]
        # pos = [map*episode*step*3]
        # idx = [map*step*3] which map, which episode, which step
        self.img = img
        self.pos = pos
        self.args = args
        self.k_size = math.ceil((self.args.epi_nums - self.args.k_test)/self.args.k_fold)
        self.model = Predictor(args.attn_heads, self.args.feature_size, self.args.embedding_size)
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.wd)
        self.train_maps = args.train_maps
        self.train_data, self.test_data = self.mem_process(zone_data)
        self.sim = sim(dim=-1, eps=self.args.eps)
        if torch.cuda.is_available():
            self.model.cuda()
            self.sim.cuda()
        self.temperature = args.temperature
    
    def k_fold_generator(self, k_idx):
        k_range = [k_idx*self.k_size, (k_idx+1)*self.k_size]
        train_slice = {}
        val_slice = {}
        for key in self.train_data.keys():
            mapid, epi, zone_id = key.split('-')
            epi = int(epi)
            if not (epi >= k_range[0] and epi < k_range[1]):
                train_slice[key] = self.train_data[key]
            else:
                val_slice[key] = self.train_data[key]
        train_data = myData(train_slice)
        val_data = myData(val_slice)
        train_data = myLoader(train_data, batch_size=self.args.batch_size,shuffle=True)
        val_data = myLoader(val_data, batch_size=len(list(val_data.zone.keys())))
        return train_data, val_data

    def mem_process(self, inp):
        temp = {}
        temp_test = {}
        for key in inp.keys():
            mapid, epi, zone = key.split('-')
            mapid = int(mapid)
            epi = int(epi)
            if mapid in self.train_maps:
                if epi + self.args.k_test >= self.args.epi_nums:
                    temp_test[key] = inp[key]
                else:
                    temp[key] = inp[key]
        return temp, temp_test

    def loss(self, predict, gt_predict, target_key):
        upper = self.sim(predict, gt_predict)
        upper = torch.exp(upper/self.args.temperature)
        idx = 0
        temp = torch.zeros([(len(self.train_maps))*self.args.zone_nums*(self.args.epi_nums-self.args.k_test) - 1, self.args.feature_size])
        for key in self.train_data.keys():
            if key != target_key:
                loc = self.train_data[key][random.randint(0,len(self.train_data[key])-1)]
                temp[idx] = torch.tensor(self.img[loc[0], loc[1], loc[2]])
        if torch.cuda.is_available():
            temp = temp.cuda()
        lower = self.sim(predict, temp)
        lower = torch.sum(torch.exp(lower/self.args.temperature))
        loss = -torch.log(upper/lower)
        temp = None
        return loss
            

    def min_element(self, inp):
        ans = {}
        for key in inp.keys():
            min_step = 1000
            for i in range(len(inp[key])):
                if inp[key][i][2] < min_step:
                    min_step = inp[key][i][2]
                    temp = inp[key][i]
            ans[key] = temp
        return ans

    def train(self):
        if self.args.model_path is not None:
            self.model.load_state_dict(torch.load(self.args.model_path))
            self.total_results = None
            return

        total_results = []
        for k in range(self.args.k_fold):
            train, val = self.k_fold_generator(k)
            self.model.train()
            for epoch in range(self.args.epoch):
                epoch_loss = []
                epoch_average_acc = []
                for loc in train:
                    loss = 0
                    batch_samples = self.min_element(loc)
                    for key in batch_samples.keys():
                        mapid, epi, step, _ = batch_samples[key]
                        if step == 0:
                            step = 1 #TODO
                        train_img = self.img[mapid, epi, :step]
                        train_pos = self.pos[mapid, epi, :step]
                        q_loc = loc[key][random.randint(0,len(loc[key])-1)]
                        q_pos = self.pos[q_loc[0], q_loc[1], q_loc[2]]
                        q_feature = torch.tensor(self.img[q_loc[0], q_loc[1], q_loc[2]])
                        if torch.cuda.is_available():
                            train_img = torch.tensor(train_img).float().cuda()
                            train_pos = torch.tensor(train_pos).float().cuda()
                            q_pos = torch.tensor(q_pos).float().cuda()
                            q_feature = q_feature.cuda()
                        feature = self.model(train_img, train_pos, q_pos)
                        loss += self.loss(feature, q_feature, key)
                        epoch_average_acc.append(self.sim(feature, q_feature).item())
                    loss = loss/len(list(batch_samples.keys()))
                    epoch_loss.append(loss.item())            
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                print('train epoch: {}, loss: {:.4}, acc: {:.4}'.format(epoch+1, np.mean(epoch_loss), np.mean(epoch_average_acc)))
                wandb.log({'loss':np.mean(epoch_loss), 'acc':np.mean(epoch_average_acc)})

            self.model.eval()
            with torch.no_grad():
                val_acc = []
                for loc in val:
                    batch_samples = self.min_element(loc)
                    for key in batch_samples.keys():
                        mapid, epi, step, _ = batch_samples[key]
                        if step == 0:
                            step = 1
                        train_img = self.img[mapid, epi, :step]
                        train_pos = self.pos[mapid, epi, :step]
                        q_loc = loc[key][random.randint(0,len(loc[key])-1)]
                        q_pos = self.pos[q_loc[0], q_loc[1], q_loc[2]]
                        q_feature = self.img[q_loc[0], q_loc[1], q_loc[2]]
                        if torch.cuda.is_available():
                            train_img = torch.tensor(train_img).float().cuda()
                            train_pos = torch.tensor(train_pos).float().cuda()
                            q_pos = torch.tensor(q_pos).float().cuda()
                            q_feature = torch.tensor(q_feature).float().cuda()
                        feature = self.model(train_img, train_pos, q_pos)
                        val_acc.append(self.sim(feature, q_feature).item())
                print('acc: {:.4}'.format(np.mean(val_acc)))
                wandb.log({'eval_acc':np.mean(val_acc)})
                total_results.append(np.mean(val_acc))
                print("***************************K={} FINISHED********************************".format(k+1))
        
        torch.save(self.model.state_dict(),'./model_cosine.pt')
        return total_results

    

"""class main():
    def __init__(self, args):
        self.args = args
        self.pre = pre_process(args)
        wandb.init(project='predictor',entity='ethanyang')
        config = {}
        for name, value in vars(args).items():
            config[str(name)] = value
        wandb.config = config
    
    def calcu(self):
        #img, pos = self.pre.process()
        img = np.load('data.npy')
        pos = np.load('pos.npy')
        #sys.exit()
        zone_data = self.pre.clust()
        self.train = trainer(self.args, img, pos, zone_data)
        results = self.train.train()
        ic(results)

args = Args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
pro = main(args)
pro.calcu()"""