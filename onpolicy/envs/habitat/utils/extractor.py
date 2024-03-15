import math
import os

import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torchvision
from PIL import Image
import onpolicy
from icecream import ic

# --------------------------------------
# Pooling layers
# --------------------------------------

class MAC(nn.Module):

    def __init__(self):
        super(MAC,self).__init__()

    def mac(self, x):
        return F.max_pool2d(x, (x.size(-2), x.size(-1)))

    def forward(self, x):
        return self.mac(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class SPoC(nn.Module):

    def __init__(self):
        super(SPoC,self).__init__()

    def spoc(self, x):
        return F.avg_pool2d(x, (x.size(-2), x.size(-1)))

    def forward(self, x):
        return self.spoc(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class GeMmp(nn.Module):

    def __init__(self, p=3, mp=1, eps=1e-6):
        super(GeMmp,self).__init__()
        self.p = Parameter(torch.ones(mp)*p)
        self.mp = mp
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def forward(self, x):
        return self.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '[{}]'.format(self.mp) + ', ' + 'eps=' + str(self.eps) + ')'

class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC,self).__init__()
        self.L = L
        self.eps = eps

    def rmac(self, x, L=3, eps=1e-6):
        ovr = 0.4 # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w/2.0 - 1)

        b = (max(H, W)-w)/(steps-1)
        (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:  
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

        for l in range(1, L+1):
            wl = math.floor(2*w/(l+1))
            wl2 = math.floor(wl/2 - 1)

            if l+Wd == 1:
                b = 0
            else:
                b = (W-wl)/(l+Wd-1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
            if l+Hd == 1:
                b = 0
            else:
                b = (H-wl)/(l+Hd-1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
                
            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                    R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                    vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                    vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                    v += vt

        return v

    def forward(self, x):
        return self.rmac(x, L=self.L, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

class Rpool(nn.Module):
    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super(Rpool,self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def roipool(self, x, rpool, L=3, eps=1e-6):
        ovr = 0.4 # desired overlap of neighboring regions
        steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w/2.0 - 1)

        b = (max(H, W)-w)/(steps-1)
        _, idx = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:  
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        vecs = []
        vecs.append(rpool(x).unsqueeze(1))

        for l in range(1, L+1):
            wl = math.floor(2*w/(l+1))
            wl2 = math.floor(wl/2 - 1)

            if l+Wd == 1:
                b = 0
            else:
                b = (W-wl)/(l+Wd-1)
            cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b).int() - wl2 # center coordinates
            if l+Hd == 1:
                b = 0
            else:
                b = (H-wl)/(l+Hd-1)
            cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b).int() - wl2 # center coordinates
                
            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    vecs.append(rpool(x.narrow(2,i_,wl).narrow(3,j_,wl)).unsqueeze(1))

        return torch.cat(vecs, dim=1)

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = self.roipool(x, self.rpool, self.L, self.eps) # size: #im, #reg, D, 1, 1

        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0]*s[1], s[2], s[3], s[4]) # size: #im x #reg, D, 1, 1

        # rvecs -> norm
        o = self.norm(o)

        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))

        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4]) # size: #im, #reg, D, 1, 1

        # aggregate regions into a single global vector per image
        if aggregate:
            # rvecs -> sumpool -> norm
            o = self.norm(o.sum(1, keepdim=False)) # size: #im, D, 1, 1

        return o

    def __repr__(self):
        return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'model/encoders/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'model/encoders/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'model/encoders/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'model/encoders/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'model/encoders/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'model/encoders/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'model/encoders/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'model/encoders/gl18-tl-resnet152-gem-w-21278d5.pth',
}

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'model/encoders/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'model/encoders/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'model/encoders/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'model/encoders/imagenet-caffe-resnet152-features-1011020.pth',
}

# TODO: pre-compute for more architectures and properly test variations (pre l2norm, post l2norm)
# pre-computed local pca whitening that can be applied before the pooling layer
L_WHITENING = {
    'resnet101' : 'model/encoders/retrieval-SfM-120k-resnet101-lwhiten-9f830ef.pth', # no pre l2 norm
    # 'resnet101' : 'model/encoders/retrieval-SfM-120k-resnet101-lwhiten-da5c935.pth', # with pre l2 norm
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac'   : MAC,
    'spoc'  : SPoC,
    'gem'   : GeM,
    'gemmp' : GeMmp,
    'rmac'  : RMAC,
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    'alexnet-gem-r'   : 'model/encoders/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth',
    'vgg16-gem-r'     : 'model/encoders/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth',
    'resnet101-mac-r' : 'model/encoders/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth',
    'resnet101-gem-r' : 'model/encoders/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth',
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    'alexnet-gem'            : 'model/encoders/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'alexnet-gem-r'          : 'model/encoders/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth',
    'vgg16-gem'              : 'model/encoders/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'vgg16-gem-r'            : 'model/encoders/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth',
    'resnet50-gem'           : 'model/encoders/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
    'resnet101-mac-r'        : 'model/encoders/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth',
    'resnet101-gem'          : 'model/encoders/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet101-gem-r'        : 'model/encoders/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth',
    'resnet101-gemmp'        : 'model/encoders/retrieval-SfM-120k-resnet101-gemmp-whiten-770f53c.pth',
    'resnet152-gem'          : 'model/encoders/retrieval-SfM-120k-resnet152-gem-whiten-abe7b93.pth',
    'densenet121-gem'        : 'model/encoders/retrieval-SfM-120k-densenet121-gem-whiten-79e3eea.pth',
    'densenet169-gem'        : 'model/encoders/retrieval-SfM-120k-densenet169-gem-whiten-6b2a76a.pth',
    'densenet201-gem'        : 'model/encoders/retrieval-SfM-120k-densenet201-gem-whiten-22ea45c.pth',
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg11'                 :  512,
    'vgg13'                 :  512,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              :  512,
    'resnet34'              :  512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'densenet121'           : 1024,
    'densenet169'           : 1664,
    'densenet201'           : 1920,
    'densenet161'           : 2208, # largest densenet
    'squeezenet1_0'         :  512,
    'squeezenet1_1'         :  512,
}

class VisualEncoder():

    def __init__(self, args):
        self.args = args
        self.is_blind = False
        if args.encoder_url is not None:

            print(">> Loading encoder from:\n>>>> '{}'".format(args.encoder_url))
            if args.encoder_url in PRETRAINED:
                # pretrained networks (downloaded automatically)
                state = torch.load(onpolicy.__path__[0]+ "/envs/habitat/" +PRETRAINED[args.encoder_url])
            else:
                raise NotImplementedError 
            # parsing net params from meta
            # architecture, pooling, mean, std required
            # the rest has default values, in case that is doesnt exist
            net_params = {}
            net_params['architecture'] = state['meta']['architecture']
            net_params['pooling'] = state['meta']['pooling']
            net_params['local_whitening'] = state['meta'].get('local_whitening', False)
            net_params['regional'] = state['meta'].get('regional', False)
            net_params['whitening'] = state['meta'].get('whitening', False)
            net_params['mean'] = state['meta']['mean']
            net_params['std'] = state['meta']['std']
            net_params['pretrained'] = False

            # load network
            net = init_network(net_params)
            net.load_state_dict(state['state_dict'])

            # if whitening is precomputed
            if 'Lw' in state['meta']:
                net.meta['Lw'] = state['meta']['Lw']
            
            print(">>>> loaded network: ")
            print(net.meta_repr())
        
        else:
            raise NotImplementedError
        
        if self.args.cuda:
            net.cuda()

        net.eval()
        self.net = net

        # set up the transform
        self.normalize = transforms.Normalize(
            mean=net.meta['mean'],
            std=net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])


        # compute whitening
        if args.whitening is not None:

            if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
                
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))

                Lw = net.meta['Lw'][args.whitening]['ss']

            else:
                if args.encoder_url is not None:
                    whiten_fn = args.encoder_url + '_{}_whiten'.format(args.whitening)

                    whiten_fn += '.pth'
                else:
                    whiten_fn = None

                if whiten_fn is not None and os.path.isfile(whiten_fn):
                    print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                    Lw = torch.load(whiten_fn)
                else:
                    raise NotImplementedError

        else:
            Lw = None
        
        self.Lw = Lw

    def extract(self, query):
        query = query[:,:3].cpu()
        vec = []
        for i in range(query.shape[0]):
            img = transforms.ToPILImage()(query[i].squeeze())
            img.thumbnail((self.args.img_resol, self.args.img_resol), Image.ANTIALIAS)
            query[i] = self.transform(img)

        if self.args.cuda:
            query = query.cuda()
        vec = extract_ss(self.net, query).cpu().detach()
        
        if len(vec.shape) == 1:
            vec = vec.unsqueeze(-1)
        
        dtype = vec.dtype
        if self.Lw is not None:
            # whiten the vectors
            vec  = whitenapply(vec, self.Lw['m'], self.Lw['P'])

        vec = vec.permute(1,0).to(dtype)
        if self.args.cuda:
            vec = vec.cuda()
        return vec




class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(_get_data_root(), 'whiten')
                lwhiten.load_state_dict(torch.load(onpolicy.__path__[0]+ "/envs/habitat/" +L_WHITENING[lw]))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    if pooling == 'gemmp':
        pool = POOLING[pooling](mp=dim)
    else:
        pool = POOLING[pooling]()
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(_get_data_root(), 'whiten')
                rwhiten.load_state_dict(torch.load(onpolicy.__path__[0]+ "/envs/habitat/" +R_WHITENING[rw]))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening

    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(_get_data_root(), 'whiten')
                whiten.load_state_dict(torch.load(onpolicy.__path__[0]+ "/envs/habitat/" +WHITENING[w]))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(_get_data_root(), 'networks')
        net.features.load_state_dict(torch.load(onpolicy.__path__[0]+ "/envs/habitat/" +FEATURES[architecture]))

    return net

def extract_ss(net, input):
    return net(input).squeeze()


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def l2n(self, x, eps=1e-6):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def forward(self, x):
        return self.l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

def _get_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

def _get_data_root():
    return os.path.join(_get_root(), 'data')

def whitenapply(X, m, P, dimensions=None):
    m = torch.tensor(m)

    if not dimensions:
        dimensions = P.shape[0]

    X = np.dot(P[:dimensions, :], X-m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)

    return torch.tensor(X)