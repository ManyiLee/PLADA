import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn import functional as F
from timm.layers import trunc_normal_, DropPath

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = F.linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


class Multi_head_attention_forward(nn.Module):
    def __init__(self, dim, design_details, _CG_tuning = False, num_heads = 16,act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=None):
        super().__init__()
        
        
        if _CG_tuning:
            # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
            dpr , drop_rate, mlp_ratio = design_details['CG']['dpr'] , design_details['CG']['drop_rate'], design_details['CG']['mlp_ratio'], 
            num_tokens, prompt_length, comp_out = design_details['num_tokens'], design_details['prompt_length'], design_details['CG']['comp_out']
            self.norm = norm_layer(dim)
            self.drop_path1 = DropPath(dpr) if dpr > 0. else nn.Identity()
            self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_rate)
            
            self.global_comp = nn.Conv1d(num_tokens + prompt_length, comp_out, kernel_size=1)
            nn.init.uniform_(self.global_comp.weight.data, -1, 1)

            self.local_comps = nn.ModuleList([nn.Conv1d(num_tokens + prompt_length, comp_out, kernel_size=1) for _ in range(num_heads)])
            for l_comp in self.local_comps:
                nn.init.uniform_(l_comp.weight.data, -1, 1)
            
        self._CG_tuning = _CG_tuning
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        prefix: Tensor = None,
        batch_weight: Tensor = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape  # for visual: [259, bs, 1024]; for text: [77, n_cls, 512]
        src_len, _, _ = key.shape
        if prefix is not None:
            _, _, prefix_len, _ = prefix.shape  # [bs, 2, prefix_len, embed_dim]

        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        # compute in-projection
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        if prefix is not None:
            prefix_k = prefix[:, 0, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]
            prefix_v = prefix[:, 1, ...].transpose(0, 1).contiguous()  # [prefix_len, bs, embed_dim]
            
        # prep attention mask
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
            
        # reshape q, k, v for multihead attention and make em batch first
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # for visual: [bs*16, 259, 64]; for text: [n_cls*8, 77, 64]
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        
        if prefix is not None:
            prefix_k = prefix_k.view(prefix_k.size(0), bsz * num_heads, head_dim).transpose(0, 1)  #for visual: [bs*16, prefix_len, 64]; for text: [n_cls*8, prefix_len, 64]
            prefix_v = prefix_v.view(prefix_v.size(0), bsz * num_heads, head_dim).transpose(0, 1)
            
            
        # (deep breath) calculate attention and out projection
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)  # only for text: [1, 1, 77, 77]
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)  # for visual: [bs, 16, 259, 64]; for text: [n_cls, 8, 77, 64]
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)
        
        if prefix is not None:
            prefix_k = prefix_k.view(bsz, num_heads, prefix_len, head_dim)  #for visual: [bs, 16, prefix_len, 64]; for text: [n_cls, 8, prefix_len, 64]
            prefix_v = prefix_v.view(bsz, num_heads, prefix_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)  # for visual: [bs, 16, 259, 64]; for text: [n_cls, 8, 77, 64]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  # for visual: [259*bs, 1024]; for text: [77*n_cls, 512]
        
        
        if prefix is not None:
            
            # this operation called Residual Guidance
            attn_output_prefix = F.scaled_dot_product_attention(q, prefix_k, prefix_v, None, dropout_p, is_causal)  # for visual: [bs, 12, 197, 64]; for text: [n_cls, 8, 77, 64]
            if batch_weight is not None:
                attn_output_prefix = attn_output_prefix * batch_weight.view(bsz, 1, 1, 1)
            attn_output_prefix = attn_output_prefix.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  # for visual: [197*bs, 768]; for text: [77*n_cls, 512]
            attn_output += attn_output_prefix
            #============ first attention shifting finished ==============#
            
            
            if self._CG_tuning: #further fusion, in shallow layers generally
                
                head_attentions = attn_output.view(bsz, tgt_len, num_heads, head_dim)
                latent_feat = query.permute(1, 0, 2) + self.drop_path1(self.ls1(attn_output.view(bsz, tgt_len, embed_dim))) #[bsz, 259, 1024]
                mlp_feat =  self.mlp(self.norm(latent_feat)) #[bsz, 259, 1024]
                
                global_feat = torch.mean(self.global_comp(mlp_feat), dim = 1).squeeze(1) # (bsz, 259->128, dim) --> (bsz, dim)
                
                head_attentions = head_attentions.permute(2, 0, 1, 3)
                head_feats = []
                for head_attn, local_comp in zip(head_attentions, self.local_comps):
                    head_feats.append(torch.mean(local_comp(head_attn), dim=1).squeeze(1)) # (bsz, 259->128, dim) --> (bsz, Hdim)
                local_feat = torch.cat(head_feats, dim=0).reshape(bsz,-1) #[heads, bsz, Hdim] --> (bsz, dim)
                
                #[prefix_len + tgt_len, bs, embed_dim]
                xk = torch.cat([prefix[:, 0, ...].transpose(0, 1).contiguous() * local_feat, query],dim=0)
                xv = torch.cat([prefix[:, 1, ...].transpose(0, 1).contiguous() * global_feat, query],dim=0)

                xk = xk.view(xk.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
                xv = xv.view(xv.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
                
                xk = xk.view(bsz, num_heads, src_len + prefix_len, head_dim)
                xv = xv.view(bsz, num_heads, src_len + prefix_len, head_dim)
                
                attn_output_full = F.scaled_dot_product_attention(q, xk, xv, None, dropout_p, is_causal)
                attn_output_full = attn_output_full.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
                #============ second attention shifting finished ==============#
                attn_output = attn_output + attn_output_full
        
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # for visual: [259*bs, 1024]; for text: [77*n_cls, 512]
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))  # for visual: [259, bs, 1024]; for text: [77, n_cls, 512]
        
        return attn_output, None


class MultiheadAttention(Module):
    
    def __init__(self, embed_dim, num_heads, design_details, _CG_tuning = False, device=None, dtype=None, prefix_pool_size=0, prefix_len=0) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

        self.add_preifx = False
        if prefix_pool_size > 0:
            self.add_preifx = True
            prefix_shape = (prefix_pool_size, 2, prefix_len, embed_dim)
            prefix_pool = torch.zeros(prefix_shape, dtype=torch.float16)
            torch.nn.init.uniform_(prefix_pool[:, 0], -1, 1)
            self.prefix_pool = Parameter(prefix_pool)
        
        self.multi_head_attention_forward = Multi_head_attention_forward(
                                                                        dim=embed_dim,
                                                                        design_details = design_details,
                                                                        _CG_tuning = _CG_tuning,
                                                                        num_heads = num_heads)
        self.design_details = design_details
        
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            prompt_ids: Tensor = None,
            batch_weight: Tensor = None
            ) -> Tuple[Tensor, Optional[Tensor]]:

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        prefix = None
        if self.add_preifx:
            assert prompt_ids.size(1) == 1, "Only single prefix for one sample is supported."
            prompt_ids = prompt_ids.squeeze(1)
            if prompt_ids.size(0) == 1:
                prompt_ids = prompt_ids.repeat(query.size(1))
            #================Choose Prompt===============#
            if self.training or self.design_details['Test_Prompt_AVG'] == "No":
                prefix = self.prefix_pool[prompt_ids]  # [bs, 2, prefix_len, embed_dim], one for key, another for value
            elif self.design_details['Test_Prompt_AVG'] == "Half":
                # half of AVG, one for real , other for fake
                pool_size = self.prefix_pool.shape[0]
                prefix_avg = torch.concat([self.prefix_pool[:pool_size//2].mean(dim=0,keepdim=True), self.prefix_pool[pool_size//2:].mean(dim=0, keepdim=True)],dim = 0)
                prefix = prefix_avg[prompt_ids]
            elif self.design_details['Test_Prompt_AVG'] == "Full":
                prefix = self.prefix_pool.mean(dim=0,keepdim=True).repeat(prompt_ids.shape[0], 1, 1 , 1)
            else:
                raise RuntimeError("We don't support other methods for taclking prompts when test, please type No、Half、Full.")
                
        attn_output, attn_output_weights = self.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            None, None, False,
            0., self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            attn_mask=attn_mask, 
            prefix=prefix, 
            batch_weight=batch_weight
            )
        return attn_output, attn_output_weights