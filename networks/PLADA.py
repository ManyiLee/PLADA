import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np

from torch.autograd import Function
from networks.resnet import resnet50
from networks.hkr import *
from models import get_model

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class ReverseLayer(Function):
    """
    Reverse Layer component
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Conv2d1x1(nn.Module):
    """
    self-attention mechanism: score map * feature
    """
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.fc1 = nn.Linear(in_f, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_f)
        self.fc3 = nn.Linear(in_f, out_f)
        self.lkrelu = nn.LeakyReLU()

    def forward(self, x):
        
        att = x #[bs, in_f]

        att1 = self.fc1(att) #[bs, hd]
        att2 = self.lkrelu(att1) #[bs, hd]
        score_map = self.fc2(att2) #[bs,out_f]
        score_map = F.softmax(score_map, dim = -1)

        out = self.fc3(x) #[bs, out_f]
        attention = torch.mul(score_map, out)  

        x = out + attention
        x = self.lkrelu(x)

        return x

class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()
        #self.do = nn.Dropout(0.2)
        self.mlp = nn.Sequential(nn.Linear(in_f, out_f))

    def forward(self, x):
        #bs = x.size()[0]
        #x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x)
        #x = self.do(x)
        return x #, x_feat


class PLADA(nn.Module):
    def name(self):
        return 'PLADA'

    def __init__(self, opt):
        super(PLADA, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt['isTrain']
        self.save_dir = os.path.join(opt['checkpoints_dir'], opt['name'])
        self.device = torch.device('cuda')
        self.B2E = self.opt["B2E"]

        self.encoder_feat_dim = 768
        self.num_classes = 1
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.head_cmp = Head(
            in_f=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_tf = Head(
            in_f=self.encoder_feat_dim,
            out_f=self.num_classes
        )

        self.block_cmp = Conv2d1x1(
            in_f = self.encoder_feat_dim,
            hidden_dim=self.encoder_feat_dim // 2, 
            out_f=self.encoder_feat_dim
        )
        self.block_tf = Conv2d1x1(
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.encoder_feat_dim // 2, 
            out_f=self.encoder_feat_dim 
        )
        
        if self.isTrain and not opt['continue_train']:
            self.backbone = get_model('CLIP:ViT-L/14', self.B2E)
            
        if not self.isTrain or opt['continue_train']:
            self.backbone = get_model('CLIP:ViT-L/14', self.B2E)

        if len(opt['device_ids']) > 1:
            self.backbone = torch.nn.DataParallel(self.backbone)

    def forward(self, input, train = False, label = None, alpha = None):
        
        #=================Choose Prompt=================#
        
        # we have three ways to choose Prompt
        # for taining
        # 1. Random Selection
        # 2. Beam Selection
        
        # for inference
        # 1. Random Selection
        # 2. Full Average
        # 3. Half Average
        
        bsz = input.shape[0]
        if train:
            tf_label_np, mask_label_np = label
            if self.B2E['Train_random']:
                # Random Selection
                prompt_ids = torch.randint(low=0, high=self.B2E['pool_size'], size=(bsz, 1), requires_grad=False).cuda()
            else:
                # Beam Selection
                prompt_ids_fake = torch.tensor(np.random.randint(self.B2E['pool_size'] // 2,self.B2E['pool_size'],bsz) & tf_label_np).view(bsz, 1)
                prompt_ids_real = torch.tensor(np.random.randint(0,self.B2E['pool_size']//2,bsz) & ~tf_label_np).view(bsz, 1)
                prompt_ids = (prompt_ids_fake + prompt_ids_real).cuda()
                prompt_ids.requires_grad_(False)
        else:
            if self.B2E['Test_Prompt_AVG'] == "No":
                # Random Selection
                prompt_ids = torch.randint(low=0, high=self.B2E['pool_size'], size=(input.shape[0], 1), requires_grad=False).cuda()
            elif self.B2E['Test_Prompt_AVG'] in ["Half","Full"] :
                # Full or Half Average
                prompt_ids = torch.randint(low=0, high=2, size=(input.shape[0], 1), requires_grad=False).cuda()
            else:
                raise RuntimeError("We don't support other methods for taclking prompts when test, please tye No、Half、Full.")
            
            
        backbone_feat = self.backbone(input, prompt_ids)
        tf_feat = self.block_tf(backbone_feat)
        out_tf = self.head_tf(tf_feat)
        tf_loss, cmp_loss, dis_loss = None, None, None
        if train:
            tf_label = torch.tensor(tf_label_np).float().cuda()
            tf_loss = self.loss_fn(out_tf.squeeze(-1), tf_label)

            if sum(mask_label_np) != 0:
                
                #reverse branch
                if alpha is not None:
                    reverse_feat = backbone_feat[mask_label_np]
                    reverse_feat = ReverseLayer.apply(reverse_feat, alpha)
                    cmp_feat = self.block_cmp(reverse_feat)
                    out_cmp = self.head_cmp(cmp_feat)
                    
                    cmp_label = np.zeros(sum(mask_label_np)).astype(bool)
                    cmp_label[sum(mask_label_np) // 2:] = True
                    cmp_label = torch.tensor(cmp_label).float().cuda()
                    cmp_loss = self.loss_fn(out_cmp.squeeze(-1), cmp_label)
                else:
                    cmp_loss = None
                
            
            #TF DIS            
            f_cmp, f_no_cmp = tf_feat[tf_label_np&~mask_label_np], tf_feat[tf_label_np&~mask_label_np]
            t_cmp, t_no_cmp =  tf_feat[~tf_label_np&~mask_label_np], tf_feat[~tf_label_np&~mask_label_np]

            # center of no cmp images
            FNCC = f_no_cmp.mean(dim=0, keepdim=True)  
            TNCC = t_no_cmp.mean(dim=0, keepdim=True)
            
            #center of cmp images
            FCC = f_cmp.mean(dim=0, keepdim=True)  
            TCC = t_cmp.mean(dim=0, keepdim=True)

            if torch.isnan(FNCC).sum() != 0 or torch.isnan(TNCC).sum() != 0 or torch.isnan(FCC).sum() != 0 or \
                torch.isnan(TCC).sum() != 0:
                print("DisFeat exists None")
            else:
                if self.opt['ODA']['dist'] == 'L1':
                    dis_nc = 1.0 / (1 + torch.abs(FNCC - TNCC).sum())
                    dis_c = 1.0 / (1 + torch.abs(FCC - TCC).sum())
                elif self.opt['ODA']['dist'] == 'L2':
                    dis_nc = 1.0 / (1 + torch.sqrt(torch.pow(FNCC - TNCC, 2).sum()))
                    dis_c = 1.0 / (1 + torch.sqrt(torch.pow(FCC - TCC, 2).sum()))
                elif self.opt['ODA']['dist'] == 'cosin':
                    dis_nc = 1.0 / (1 - torch.cosine_similarity(FNCC, TNCC).item())
                    dis_c = 1.0 / (1 - torch.cosine_similarity(FCC, TCC).item())
                elif self.opt['ODA']['dist'] == 'KL':
                    KL_nc = 0.5 * F.kl_div(FNCC.softmax(dim=-1).log(), TNCC.softmax(dim=-1), reduction='sum') \
                            + 0.5 * F.kl_div(TNCC.softmax(dim=-1).log(), FNCC.softmax(dim=-1), reduction='sum')
                    KL_c = 0.5 * F.kl_div(FCC.softmax(dim=-1).log(), TCC.softmax(dim=-1), reduction='sum') \
                            + 0.5 * F.kl_div(TCC.softmax(dim=-1).log(), FCC.softmax(dim=-1), reduction='sum')
                    dis_nc = 1.0 / (1 + KL_nc)
                    dis_c = 1.0 / (1 + KL_c)
                elif self.opt['ODA']['dist'] == 'JS':
                    mid_nc = (FNCC + TNCC) / 2
                    mid_c = (FCC + TCC) / 2
                    JS_nc = 0.5 * F.kl_div(FNCC.softmax(dim=-1).log(), mid_nc.softmax(dim=-1), reduction='sum') \
                            + 0.5 * F.kl_div(TNCC.softmax(dim=-1).log(), mid_nc.softmax(dim=-1), reduction='sum')
                    JS_c = 0.5 * F.kl_div(FCC.softmax(dim=-1).log(), mid_c.softmax(dim=-1), reduction='sum') \
                            + 0.5 * F.kl_div(TCC.softmax(dim=-1).log(), mid_c.softmax(dim=-1), reduction='sum')
                    dis_nc = 1.0 / (1 +JS_nc)
                    dis_c = 1.0 / (1 + JS_c)
                else:
                    raise RuntimeError(f"We dont't don't support {self.opt['ODA']['dist']}, sorry...")
               
                dis_loss = dis_nc + dis_c
        
            return tf_loss, cmp_loss, dis_loss, out_tf
        else:
            
            return out_tf


    def save_networks(self, name, epoch, optimizer):
        save_filename = 'model_epoch_%s.pth' % name
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'epoch':epoch, 
            'model': self.state_dict(),
            'total_steps' : self.total_steps,
            'optimizer': optimizer.state_dict()
        }

        torch.save(state_dict, save_path)

    def adjust_learning_rate(self, optimizer ,min_lr=1e-6):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True
