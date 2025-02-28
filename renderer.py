# -*- coding: utf-8 -*-
"""code for ray marching and signal rendering
"""
import torch
import numpy as np
import torch.nn.functional as F
import scipy.constants as sc
from einops import rearrange, repeat


class Renderer():

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """

        ## Rendering parameters
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']


    def sample_points(self, rays_o, rays_d):
        """sample points along rays

        Parameters
        ----------
        rays_o : tensor. [n_rays, 3]. The origin of rays
        rays_d : tensor. [n_rays, 3]. The direction of rays

        Returns
        -------
        pts : tensor. [n_rays, n_samples, 3]. The sampled points along rays
        t_vals : tensor. [n_rays, n_samples]. The distance from origin to each sampled point
        """
        shape = list(rays_o.shape)
        shape[-1] = 1
        near, far = torch.full(shape, self.near), torch.full(shape, self.far)
        t_vals = torch.linspace(0., 1., steps=self.n_samples) * (far - near) + near  # scale t with near and far
        t_vals = t_vals.to(rays_o.device)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]

        return pts, t_vals




class Renderer_spectrum(Renderer):
    """Renderer for spectrum (integral from single direction)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)



    def render_ss(self, tx, rays_o, rays_d):
        """render the signal strength of each ray

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 3]. The direction of rays
        """

        # sample points along rays
        #每个points就是每天射线上经过的每个voxe: pts: [batchsize, n_points, 3]; t_vals: [batchsize, n_points]这个是每个点到原点的距离
        pts, t_vals = self.sample_points(rays_o, rays_d) 

        # Expand views and tx to match the shape of pts
        view = rays_d[:, None].expand(pts.shape) # [batchsize, n_points, 3],这个就是每个点的方向,就是文章中的w，因为一个ray上的所有点的方向都是一样的，所以直接相当于复制了n_points次作为每个点的w
        tx = tx[:, None].expand(pts.shape)       # [batchsize, n_points, 3]

        # Run network and compute outputs
        raw = self.network_fn(pts, view, tx)    # [batchsize, n_samples, 4]: 即attn_amp, attn_phase, signal_amp, signal_phase
        receive_ss = self.raw2outputs(raw, t_vals, rays_d)  # [batchsize]

        return receive_ss


    #将模型的原始预测转换为最终的接收到的信号输出，就是收到的信号是所有ray上的信号的叠加
    def raw2outputs(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values. (core part)

        Parameters
        ----------
        raw : [batchsize, n_samples, 4]. Prediction from model. 即attn_amp, attn_phase, signal_amp, signal_phase
        r_vals : [batchsize, n_samples]. Integration distance.  [batchsize, n_points]这个是每个点到原点的距离
        rays_d : [batchsize, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """
        #定义幅度衰减和相位翻转的计算函数
        raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists) #这里没问题，下面计算att_i的时候用1减回去了；cv里面alpha是是1-exp(-density),这个density是在RF里面就是attenuation
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw*dists

        #计算采样点间的距离 dists: [batchsize, n_samples],其实这里面已经包含了场景中的最大距离D的信息
        dists = r_vals[...,1:] - r_vals[...,:-1] #计算相邻采样点的间距
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [N_rays, N_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2, torch.sigmoid(s_p)*np.pi*2      #将预测的值映射到0-2pi：过一个sigmoid乘以2pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))              #将预测的值映射到0-正无穷，其实output head没过激活函数，所以这里过一下
        # att_a, s_a = torch.sigmoid(att_a), torch.sigmoid(s_a)

        alpha = raw2alpha(att_a, dists)  # [N_rays, N_samples] 计算每个ray上每个点的幅度衰减
        phase = raw2phase(att_p, dists)  # [N_rays, N_samples] 计算每个ray上每个点的相位翻转

        #把每个点当成新的tx,其射出的ray在途径的点上会累积衰减，计算每个ray上的每个点累积后的的幅度衰减和相位翻转；
        #即公式中对attenuation从0～r的积分,每个ray上的每个点都有自己的r,即到rx的距离
        #这里计算了：1. att_i和phase_i分别是每个ray上每个点材质造成的幅度衰减和相位翻转的累积值；2. path_loss是每个ray上每个点到rx的距离造成的衰减
        att_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]  #这里1 - alpha就是我们要的attenuation
        path = torch.cat([r_vals[...,1:], torch.Tensor([1e10]).cuda().expand(r_vals[...,:1].shape)], -1)             # [N_rays, N_samples]
        path_loss = 0.025 / path #根据free spce loss公式计算的920MHz就这个值，即lamda/(4*pi*d)
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        phase_i = torch.cumsum(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), phase], -1), -1)[:, :-1]
        phase_i = torch.exp(1j*phase_i)    # [N_rays, N_samples]

        #每个ray上的所有点的信号强度叠加，得到一个ray上收到的信号，即公式中对signal从0～D的积分
        receive_signal = torch.sum(s_a*torch.exp(1j*s_p)*att_i*phase_i*path_loss, -1)  # [N_rays]
        receive_signal = abs(receive_signal) #计算每个ray上的信号强度，这样就预测出了每个ray上的信号强度，即预测的空间谱每个pixel的值，因为一个空间谱可能包含90*360个ray，这个batchaize取得每个大，所以这里的输出是一个batchsize大小的ray，并不是全部的pixel

        return receive_signal




class Renderer_RSSI(Renderer):
    """Renderer for RSSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)


    def render_rssi(self, tx, rays_o, rays_d):
        """render the RSSI for each gateway. To avoid OOM, we split the rays into chunks

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        batchsize, _ = tx.shape
        rays_d = torch.reshape(rays_d, (batchsize, -1, 3))    # [batchsize, 9x36, 3]
        chunks = 36
        chunks_num = 36 // chunks
        rays_o_chunk = rays_o.expand(chunks, -1, -1).permute(1,0,2) #[bs, cks, 3]
        tags_chunk = tx.expand(chunks, -1, -1).permute(1,0,2)        #[bs, cks, 3]
        recv_signal = torch.zeros(batchsize).cuda()
        for i in range(chunks_num):
            rays_d_chunk = rays_d[:,i*chunks:(i+1)*chunks, :]  # [bs, cks, 3]
            pts, t_vals = self.sample_points(rays_o_chunk, rays_d_chunk) # [bs, cks, pts, 3]
            views_chunk = rays_d_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
            tx_chunk = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]

            # Run network and compute outputs
            raw = self.network_fn(pts, views_chunk, tx_chunk)    # [batchsize, chunks, n_samples, 4]
            recv_signal_chunks = self.raw2outputs_signal(raw, t_vals, rays_d_chunk)  # [bs]
            recv_signal += recv_signal_chunks

        return recv_signal    # [batchsize,]


    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
        r_vals : [batchsize, chunks, n_samples]. Integration distance.
        rays_d : [batchsize,chunks, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """
        wavelength = sc.c / 2.4e9
        # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength
        raw2amp = lambda raw, dists: -raw*dists

        dists = r_vals[...,1:] - r_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [batchsize,chunks, N_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples]
        phase = raw2phase(att_p, dists)

        # att_i = torch.cumprod(torch.cat([torch.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        amp_i = torch.exp(torch.cumsum(amp, -1))            # [batchsize,chunks, N_samples]
        phase_i = torch.exp(1j*torch.cumsum(phase, -1))                # [batchsize,chunks, N_samples]

        recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i, -1)  # integral along line [batchsize,chunks]
        recv_signal = torch.sum(recv_signal, -1)   # integral along direction [batchsize,]

        return abs(recv_signal)




class Renderer_CSI(Renderer):
    """Renderer for CSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)



    def render_csi(self, uplink, rays_o, rays_d):
        """render the RSSI for each gateway.

        Parameters
        ----------
        uplink: tensor. [batchsize, 52]. The uplink CSI (26 real + 26 imag)
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        rays_d = rearrange(rays_d, 'b (v d) -> b v d', d=3)    # [bs, 9x36 x 3]--->[bs, 9x36, 3]
        batchsize, viewsize, _ = rays_d.shape
        rays_o = repeat(rays_o, 'b d -> b v d', v=viewsize)    #[bs, 3]--->[bs, 9x36, 3]
        uplink = repeat(uplink, 'b d -> b v d', v=viewsize)    #[bs, 52]--->[bs, 9x36, 52]

        pts, t_vals = self.sample_points(rays_o, rays_d) # pts: [bs, 9x36, pts, 3]; t_vals: [bs, 9x36, pts]
        views = repeat(rays_d, 'b v d -> b v p d', p=self.n_samples)  # [bs, 9x36, pts, 3]
        uplink = repeat(uplink, 'b v d -> b v p d', p=self.n_samples)  # [bs, 9x36, pts, 3]

        # Run network and compute outputs; 为啥这里把uplink当作tx的位置输入给模型？:就当作是后面那个Radiacne Network的输入改了，用uplink的csi作为输入，输出是downlink的csi
        raw = self.network_fn(pts, views, uplink)    # [batchsize, 9x36, pts, 104]
        recv_signal = self.raw2outputs_signal(raw, t_vals, rays_d)  # [bs, 26]

        return recv_signal


    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks,n_samples,  attn_output_dims+sig_output_dims]. Prediction from model.
        r_vals : [batchsize, chunks:36*9, n_samples]. Integration distance.
        rays_d : [batchsize,chunks:36*9, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize, 26]. OFDM singal of each ray
        """
        wavelength = sc.c / 2.4e9  #sc.c是光速，sc是开头引入的一个python的物理常数库
        # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength  #模型预测的相位偏差 + 距离引起的相位偏差  = att_p + 2*pi*d/wavelength
        raw2amp = lambda raw, dists: -raw*dists                        #模型预测的幅度衰减 - att_a*d

        dists = r_vals[...,1:] - r_vals[...,:-1] #[batchsize, chunks:36*9, n_samples/points-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples/points]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize, chunks, n_samples/points].

        att_a, att_p, s_a, s_p = raw[...,:26], raw[...,26:52], raw[...,52:78], raw[...,78:104]    # [batchsize,chunks:36*9,n_samples, 26]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi          #将预测的值映射到-pi~pi：过一个sigmoid乘以2pi,再减去pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        dists = dists.unsqueeze(-1) #[batchsize, chunks:36*9, n_samples/points, 1]

        amp = raw2amp(att_a, dists)     #[batchsize, chunks, n_samples/points, 26]
        phase = raw2phase(att_p, dists) #[batchsize, chunks, n_samples/points, 26]

        # att_i = torch.cumprod(torch.cat([torch.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        amp_i = torch.exp(torch.cumsum(amp, -2))            #[batchsize, chunks, n_samples/points, 26]
        phase_i = torch.exp(1j*torch.cumsum(phase, -2))     #[batchsize, chunks, n_samples/points, 26]

        recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i, -2)  # integral along line [batchsize,chunks,26]
        recv_signal = torch.sum(recv_signal, 1)                           # integral along direction [batchsize, 26]

        return recv_signal




class Renderer_fsc(Renderer):
    """Renderer for CSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)



    def render_fsc(self, tx_o, rays_o, rays_d):
        """render the RSSI for each gateway.

        Parameters
        ----------
        tx_o: tensor. [batchsize, 3]. tx的位置[x,y,z]
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        rays_d = rearrange(rays_d, 'b (v d) -> b v d', d=3)    # [bs, 9x36 x 3]--->[bs, 9x36, 3]
        batchsize, viewsize, _ = rays_d.shape
        rays_o = repeat(rays_o, 'b d -> b v d', v=viewsize)    #[bs, 3]--->[bs, 9x36, 3]
        tx_o = repeat(tx_o, 'b d -> b v d', v=viewsize)        #[bs, 30]--->[bs, 9x36, 30]

        pts, t_vals = self.sample_points(rays_o, rays_d) # pts: [bs, 9x36, pts, 3]; t_vals: [bs, 9x36, pts]
        views = repeat(rays_d, 'b v d -> b v p d', p=self.n_samples)  # [bs, 9x36, pts, 3]
        tx_o = repeat(tx_o, 'b v d -> b v p d', p=self.n_samples)  # [bs, 9x36, pts, 30]

        # Run network and compute outputs;
        raw = self.network_fn(pts, views, tx_o)    # [batchsize, 9x36, pts, 4]
        recv_signal = self.raw2outputs_signal(raw, t_vals, rays_d)  # [bs/batchsize,2]
        return recv_signal


    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks, n_samples,  attn_output_dims+sig_output_dims]. Prediction from model.
        r_vals : [batchsize, chunks:36*9, n_samples]. Integration distance.
        rays_d : [batchsize,chunks:36*9, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize, 2]. 1个csi的预测值
        """
        wavelength = sc.c / 2.4e9  #sc.c是光速，sc是开头引入的一个python的物理常数库
        # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength  #模型预测的相位偏差 + 距离引起的相位偏差  = att_p + 2*pi*d/wavelength
        raw2amp = lambda raw, dists: -raw*dists                        #模型预测的幅度衰减 - att_a*d

        dists = r_vals[...,1:] - r_vals[...,:-1] #[batchsize, chunks:36*9, n_samples/points-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples/points]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize, chunks, n_samples/points].

        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [batchsize,chunks:36*9,n_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi          #将预测的值映射到-pi~pi：过一个sigmoid乘以2pi,再减去pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        amp = raw2amp(att_a, dists)     #[batchsize, chunks, n_samples/points]
        phase = raw2phase(att_p, dists) #[batchsize, chunks, n_samples/points]

        #point/vocxel loss;每条线上每个点自己的0~r积分，就是自己这个点到rx经过的所有voxel造成的累积
        amp_i = torch.exp(torch.cumsum(amp, -1))            #[batchsize, chunks, n_samples/points]
        phase_i = torch.exp(1j*torch.cumsum(phase, -1))     #[batchsize, chunks, n_samples/points]

        #path loss,这里补的10^10就是原点到最远处的距离，包含了卫星
        path = torch.cat([r_vals[...,1:], torch.Tensor([1e10]).cuda().expand(r_vals[...,:1].shape)], -1) #[batchsize, chunks, n_samples/points]
        path_loss = wavelength/(4*np.pi)/path #[batchsize, chunks, n_samples/points]                     #[batchsize, chunks, n_samples/points]
        
        #每条线上所有点求和
        recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i*path_loss, -1)  # integral along line,每条线上所有点求和 [batchsize,chunks]
        recv_signal = torch.sum(recv_signal, 1)                                     # integral along direction,所有方向的所有线求和 [batchsize]
        # Calculate RSS and carrier phase
        rss = torch.abs(recv_signal)  # [batchsize]
        carrier_phase = torch.angle(recv_signal)  # [batchsize]

        return torch.stack([rss, carrier_phase], dim=-1)  # [batchsize, 2]



renderer_dict = {"spectrum": Renderer_spectrum, "rssi": Renderer_RSSI, "csi": Renderer_CSI, "fsc": Renderer_fsc}

# 上面这3个类分别对应了3种不同的信号预测任务，分别是spectrum, rssi, csi，其实本质一样就是输入不同，输出都可预测一个复数（幅度+相位）
# render以及叠加本质是一样的，都是对信号进行积分，但是输入的数据集不一样，所以预处理的代码不同
# 还有就是raw2outputs_signal函数的输出，ss和rssi输出取abs，csi输出保留复数本身
# 代码中csi的输出是26个OFDM信号，这个是因为CSI的数据集是26个OFDM信号，所以输出也是26个OFDM信号，即用uplink作为输入预测downlink，复数有幅度有相位
# 代码中spectrum的输出是一个复数，即用tx作为输入预测rx，复数有幅度有相位，我可以用这个