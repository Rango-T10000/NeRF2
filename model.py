# -*- coding: utf-8 -*-
"""NeRF2 NN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2me = lambda x, y : torch.mean(abs(x - y))
sig2mse = lambda x, y : torch.mean((x - y) ** 2)
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)

csi2snr_2 = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1)) ** 2 /
    torch.norm(y, dim=(1)) ** 2
)

#---------计算rss和carrier phase的预测误差---------
fsc_evaluate = lambda pred, gt: (
    torch.mean((pred[:, 0:1] - gt[:, 0:1]) ** 2, dim=1, keepdim=True),  # RSS MSE (9603, 1)
    torch.mean((torch.remainder(pred[:, 1:2] - gt[:, 1:2] + torch.pi, 2 * torch.pi) - torch.pi) ** 2, dim=1, keepdim=True),  # Carrier Phase MSE (9603, 1)
    0.5 * torch.mean((pred[:, 0:1] - gt[:, 0:1]) ** 2, dim=1, keepdim=True) + 
    0.5 * torch.mean((torch.remainder(pred[:, 1:2] - gt[:, 1:2] + torch.pi, 2 * torch.pi) - torch.pi) ** 2, dim=1, keepdim=True)  # Total Error (9603, 1)
)

#均方误差：
fsc_evaluate_2 = lambda pred, gt: (
    torch.mean((pred[:, 0] - gt[:, 0]) ** 2),  # RSS MSE,返回是标量torch.Size([])
    torch.mean((torch.remainder(pred[:, 1] - gt[:, 1] + torch.pi, 2 * torch.pi) - torch.pi) ** 2),  # Carrier Phase MSE返回是标量torch.Size([])
    0.5 * torch.mean((pred[:, 0] - gt[:, 0]) ** 2) + 
    0.5 * torch.mean((torch.remainder(pred[:, 1] - gt[:, 1] + torch.pi, 2 * torch.pi) - torch.pi) ** 2)  # Total Error返回是标量torch.Size([])
)

#accuracy
fsc_accuracy = lambda pred, gt: torch.stack([
    100 * (1 - torch.abs(pred[:, 0] - gt[:, 0]) / (torch.abs(gt[:, 0]) + 1e-6)),  # RSS Accuracy
    100 * (1 - torch.abs(pred[:, 1] - gt[:, 1]) / (torch.abs(gt[:, 1]) + 1e-6))  # Carrier Phase Accuracy
], dim=1)  # Shape: (样本数, 2)









class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=3):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class NeRF2(nn.Module):

    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts':3, 'view':3, 'tx':3},
                 multires = {'pts':10, 'view':10, 'tx':10},
                 is_embeded={'pts':True, 'view':True, 'tx':False},
                 attn_output_dims=2, sig_output_dims=2):
        """NeRF2 model

        Parameters
        ----------
        D : int, hidden layer number, default by 8
        W : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        multires: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        is_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])

        ## attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        ## signal network
        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_view_dim + input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )

        ## output head, 2 for amplitude and phase
        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W//2, sig_output_dims)


    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_samples, 3], position of voxels
        view: [batchsize, n_samples, 3], view direction
        tx: [batchsize, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batchsize, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """
        # Check for NaN or Inf in pts, t_vals, views, and tx_o
        if torch.isnan(pts).any() or torch.isinf(pts).any():
            print("NaN or Inf detected in pts_before_PE:")
            print(pts)
            
        if torch.isnan(view).any() or torch.isinf(view).any():
            print("NaN or Inf detected in view_before_PE:")
            print(view)
            
        if torch.isnan(tx).any() or torch.isinf(tx).any():
            print("NaN or Inf detected in tx_before_PE:")
            print(tx)

        # position encoding; #.contiguous() 是为了保证 Tensor 在内存中的连续性，方便 view() 操作
        pts = self.embed_pts_fn(pts).contiguous() # [batchsize, n_samples/points, input_pts_dim], e.g.[batchsize, 64, 3]---->[batchsize, 64, 63]
        view = self.embed_view_fn(view).contiguous()
        tx = self.embed_tx_fn(tx).contiguous()


        # Check for NaN or Inf in pts, t_vals, views, and tx_o
        if torch.isnan(pts).any() or torch.isinf(pts).any():
            print("NaN or Inf detected in pts:")
            print(pts)
            
        if torch.isnan(view).any() or torch.isinf(view).any():
            print("NaN or Inf detected in views:")
            print(view)
            
        if torch.isnan(tx).any() or torch.isinf(tx).any():
            print("NaN or Inf detected in tx_o:")
            print(tx)


        shape = pts.shape
        pts = pts.view(-1, list(pts.shape)[-1]) # [batchsize*n_samples, input_pts_dim]
        view = view.view(-1, list(view.shape)[-1])
        tx = tx.view(-1, list(tx.shape)[-1])


        # Check for NaN or Inf in pts, t_vals, views, and tx_o
        if torch.isnan(pts).any() or torch.isinf(pts).any():
            print("NaN or Inf detected in pts_-1:")
            print(pts)
            
        if torch.isnan(view).any() or torch.isinf(view).any():
            print("NaN or Inf detected in views_-1:")
            print(view)
            
        if torch.isnan(tx).any() or torch.isinf(tx).any():
            print("NaN or Inf detected in tx_-1:")
            print(tx)

        x = pts
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts, x], -1)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in attn:")
            print(x)   

        # 在调用位置编码前检查 pts 是否为 NaN 或 Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN or Inf detected in attn:")
            print(x)    

        attn = self.attenuation_output(x)    # (batch_size*36*9*n_points, 2); (batch_size*36*9*n_points, attn_output_dims),这个输出纬度根据数据集以及任务的不同在配置文件修改
        feature = self.feature_layer(x)      # (batch_size*36*9*n_points, W); (batch_size*36*9*n_points, 256)
        x = torch.cat([feature, view, tx], -1)

        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))
        signal = self.signal_output(x)    #[batch_size*36*9*n_points, 20]; [batch_size*36*9*n_points, sig_output_dims]这个输出纬度根据数据集以及任务的不同在配置文件修改

        # 在调用位置编码前检查 pts 是否为 NaN 或 Inf
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            print("NaN or Inf detected in attn:")
            print(attn)        
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            print("NaN or Inf detected in attn:")
            print(signal)   

        outputs = torch.cat([attn, signal], -1).contiguous()    # [batchsize, n_samples, 4]
        return outputs.view(shape[:-1]+outputs.shape[-1:])
