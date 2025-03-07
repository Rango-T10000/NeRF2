# -*- coding: utf-8 -*-
"""NeRF2 runner for training and testing
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import yaml
from skimage.metrics import structural_similarity as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import scipy.io as scio

from dataloader import *
from model import *
from renderer import renderer_dict
from utils.data_painter import paint_spectrum_compare
from utils.logger import logger_config


class NeRF2_Runner():

    def __init__(self, mode, dataset_type, **kwargs) -> None:

        kwargs_path = kwargs['path']
        kwargs_render = kwargs['render']
        kwargs_network = kwargs['networks']
        kwargs_train = kwargs['train']
        self.dataset_type = dataset_type

        ## Path settings
        self.expname = kwargs_path['expname']
        self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.devices = torch.device('cuda')

        ## Logger
        log_filename = "logger.log"
        log_savepath = os.path.join(self.logdir, self.expname, log_filename)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='nerf2')
        self.logger.info("expname:%s, datadir:%s, logdir:%s", self.expname, self.datadir, self.logdir)
        self.writer = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))


        ## Networks
        self.nerf2_network = NeRF2(**kwargs_network).to(self.devices)
        params = list(self.nerf2_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(kwargs_train['lr']),
                                          weight_decay=float(kwargs_train['weight_decay']),
                                          betas=(0.9, 0.999))
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),
                                                                        last_epoch=-1)

        ## Renderer
        renderer = renderer_dict[kwargs_render['mode']]
        self.renderer = renderer(networks_fn=self.nerf2_network, **kwargs_render)
        self.scale_worldsize = kwargs_render['scale_worldsize']

        ## Print total number of parameters
        total_params = sum(p.numel() for p in params if p.requires_grad)
        self.logger.info("Total number of parameters: %s", total_params)

        ## Train Settings
        self.current_iteration = 1
        self.current_epoch = 1
        if kwargs_train['load_ckpt'] or mode == 'test':
            self.load_checkpoints()
        self.batch_size = kwargs_train['batch_size']
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']

        ## Dataset
        self.dataset = dataset_dict[dataset_type]  #这里决定是dataset是dataloader.py中的Class中的那一个实例化
        self.train_index = os.path.join(self.datadir, "train_index.txt")
        self.test_index = os.path.join(self.datadir, "test_index.txt")
        # if not os.path.exists(train_index) or not os.path.exists(test_index):
        #     split_dataset(self.datadir, ratio=0.8, dataset_type=dataset_type)
        # #------------------------original version---------------------------
        # self.logger.info("Loading training set...")
        # self.train_set = dataset(self.datadir, train_index, self.scale_worldsize)
        # self.logger.info("Loading test set...")
        # self.test_set = dataset(self.datadir, test_index, self.scale_worldsize)

        # self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        # self.logger.info("Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))

        # #------------------------fsc_progressive_version---------------------
        # self.logger.info("Loading progressive training and val set...")
        # self.train_set = dataset(self.datadir, train_index, test_index, self.scale_worldsize)
        # self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # self.logger.info("Train and val set size:%d", len(self.train_set))

        #一些超参数的初始化
        self.num_scenes = 92        


    def load_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        self.logger.info('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.nerf2_network.load_state_dict(ckpt['nerf2_network_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']


    def save_checkpoint(self):
        
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))
        torch.save({
            'current_iteration': self.current_iteration,
            'nerf2_network_state_dict': self.nerf2_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname


    def train(self):
        """train the model
        """
        self.logger.info("Start training. Current epoch:%d", self.current_epoch)
        if self.dataset_type == 'fsc_progressive':
            for scene_id in range(self.num_scenes): #每次用一个scene中的前10s数据训练,后5s数据val；self.num_scenes数就是我的epoch数
                # print(f"Training scene {scene_id + 1}/{self.num_scenes}...")
                self.logger.info(f"Training scene {scene_id + 1}/{self.num_scenes}...")

                # **1. 只加载当前 scene 的前 10s 训练数据**
                train_dataset = self.dataset(self.datadir, self.train_index, self.scale_worldsize, scene_id)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

                # **2. 训练当前 scene**
                self.nerf2_network.train()
                with tqdm(total=len(self.num_scenes * self.total_iterations), desc=f"Iteration {self.current_iteration}/{self.num_scenes * self.total_iterations}") as pbar: #显示进度条
                    for iter in range(self.total_iterations):  
                        for train_input, train_label, mask in train_loader:
                            
                            output = model(train_input)
                            loss = criterion(output, train_label)

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            self.cosine_scheduler.step()
                            self.current_iteration += 1

                self.current_epoch += 1


                # **3. 评估当前 scene（使用后 5s 数据）**
                val_data = train_dataset.json_test_data  # 5s 评估数据
                val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)  

                self.nerf2_network.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_sample in val_loader:
                        val_input, val_label, val_mask = val_sample['input'], val_sample['label'], val_sample.get('mask', None)
                        val_output = model(val_input)
                        val_loss += criterion(val_output, val_label).item()

                print(f"Scene {scene_id} Val Loss: {val_loss / len(val_loader)}")

                # 进入下一个 scene
                if scene_id + 1 < self.num_scenes:
                    train_dataset.set_scene(scene_id + 1)  # 更新数据集

#---------------------------------------------
            for train_input, train_label, mask in self.train_iter:
                if self.current_iteration > self.total_iterations:
                    break
                train_input, train_label, mask = train_input.to(self.devices), train_label.to(self.devices), mask.to(self.devices)
                rays_o, rays_d, tx_o = train_input[:, :3], train_input[:,3:3+9*36*3], train_input[:, 3+9*36*3:]
                predict = self.renderer.render_fsc(tx_o, rays_o, rays_d) #[batchsize,2]
                predict_csi = mask * predict
                loss = sig2mse(predict_csi, train_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.cosine_scheduler.step()
                self.current_iteration += 1

                self.writer.add_scalar('Loss/loss', loss, self.current_iteration)
                pbar.update(1)
                pbar.set_description(f"Iteration {self.current_iteration}/{self.total_iterations}")
                pbar.set_postfix_str('loss = {:.6f}, lr = {:.6f}'.format(loss.item(), self.optimizer.param_groups[0]['lr']))

                if self.current_iteration % self.save_freq == 0:
                    ckptname = self.save_checkpoint()
                    pbar.write('Saved checkpoints at {}'.format(ckptname))     
                else:
                    break           

    def eval_network_spectrum(self):
        """test the model
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        os.makedirs(os.path.join(self.logdir, self.expname, 'pred_spectrum'), exist_ok=True)
        pred2next, gt2next = torch.zeros((0)), torch.zeros((0))
        save_img_idx = 0
        all_ssim = []
        with torch.no_grad():
            for test_input, test_label in self.test_iter:
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                rays_o, rays_d, tx_o = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:9]
                pred_spectrum = self.renderer.render_ss(tx_o, rays_o, rays_d)


                ## save predicted spectrum
                pred_spectrum = pred_spectrum.detach().cpu()
                gt_spectrum = test_label.detach().cpu()
                pred_spectrum = torch.concatenate((pred2next, pred_spectrum), dim=0)
                gt_spectrum = torch.concatenate((gt2next, gt_spectrum), dim=0)
                num_spectrum = len(pred_spectrum) // (360 * 90)
                pred2next = pred_spectrum[num_spectrum*360*90:]
                gt2next = gt_spectrum[num_spectrum*360*90:]

                for i in range(num_spectrum):
                    pred_sepctrum_i = pred_spectrum[i*360*90:(i+1)*360*90].numpy().reshape(90, 360)
                    gt_spectrum_i = gt_spectrum[i*360*90:(i+1)*360*90].numpy().reshape(90, 360)
                    pixel_error = np.mean(abs(pred_sepctrum_i - gt_spectrum_i))
                    ssim_i = ssim(pred_sepctrum_i, gt_spectrum_i, data_range=1, multichannel=False)
                    self.logger.info("Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error, ssim_i))
                    paint_spectrum_compare(pred_sepctrum_i, gt_spectrum_i,save_path=os.path.join(self.logdir, self.expname,'pred_spectrum', f'{save_img_idx}.png'))
                    all_ssim.append(ssim_i)
                    self.logger.info("Median SSIM is {:.6f}".format(np.median(all_ssim)))
                    save_img_idx += 1
                    np.savetxt(os.path.join(self.logdir, self.expname, 'all_ssim.txt'), all_ssim, fmt='%.4f')

    def eval_network_rssi(self):
        """test the model and save predicted RSSI values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        with torch.no_grad():
            with open(os.path.join(self.logdir, self.expname, "result.txt"), 'w') as f:
                for test_input, test_label in self.test_iter:
                    test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                    tx_o, rays_o, rays_d = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:]
                    predict_rssi = self.renderer.render_rssi(tx_o, rays_o, rays_d)

                    ## save predicted spectrum
                    predict_rssi = amplitude2rssi(predict_rssi.detach().cpu())
                    gt_rssi = amplitude2rssi(test_label.detach().cpu())

                    error = abs(predict_rssi - gt_rssi.reshape(-1))
                    self.logger.info("Median error:%.2f", torch.median(error))

                    # write predicted RSSI values to file
                    for i, rssi in enumerate(predict_rssi):
                        f.write("{:.2f}, {:.2f}".format(gt_rssi[i].item(), rssi.item()) + '\n')

        result = np.loadtxt(os.path.join(self.logdir,self.expname, "result.txt"), delimiter=",")
        self.logger.info("Total Median error:%.2f", np.median(abs(result[:,0] - result[:,1])))

    def eval_network_csi(self):
        """test the model and save predicted csi values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        n_bs = self.test_set.n_bs    # number of base station antennas
        n_data = len(self.test_set)  # number of test data

        all_pred_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        all_gt_csi = torch.zeros((n_data, 26), dtype=torch.complex64)
        with torch.no_grad():
            for idx, (test_input, test_label) in enumerate(self.test_iter):
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                uplink, rays_o, rays_d = test_input[:, :52], test_input[:, 52:55], test_input[:, 55:]
                predict_downlink = self.renderer.render_csi(uplink, rays_o, rays_d)  # [B, 26]
                gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                predict_downlink = self.test_set.denormalize_csi(predict_downlink)   #这里denorm了，就知道预测的真正结果
                gt_downlink = self.test_set.denormalize_csi(gt_downlink)

                all_pred_csi[idx*self.batch_size:(idx+1)*self.batch_size] = predict_downlink
                all_gt_csi[idx*self.batch_size:(idx+1)*self.batch_size] = gt_downlink

        all_pred_csi = rearrange(all_pred_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        all_gt_csi = rearrange(all_gt_csi, '(n_data n_bs) channel -> n_data n_bs channel', n_bs=n_bs)
        snr = csi2snr(all_pred_csi, all_gt_csi)
        self.logger.info("Median SNR:%.2f", torch.median(snr))

        scio.savemat(os.path.join(self.logdir, self.expname, "result.mat"), {'pred_csi': all_pred_csi.cpu().numpy(),
                                                                              'gt_csi': all_gt_csi.cpu().numpy(),
                                                                              'snr': snr.cpu().numpy()})

    def eval_network_fsc(self):
        """test the model and save predicted fsc_csi values to a file
        """
        self.logger.info("Start evaluation")
        self.nerf2_network.eval()

        n_data = len(self.test_set)  # number of test data

        all_pred_csi = torch.zeros((n_data, 2), dtype=torch.float32)
        all_gt_csi = torch.zeros((n_data, 2), dtype=torch.float32)
        with torch.no_grad(): 
            for idx, (test_input, test_label, mask) in tqdm(enumerate(self.test_iter), total=len(self.test_iter)):
                test_input, test_label, mask = test_input.to(self.devices), test_label.to(self.devices), mask.to(self.devices)
                rays_o, rays_d, tx_o = test_input[:, :3], test_input[:,3:3+9*36*3], test_input[:, 3+9*36*3:]
                predict = self.renderer.render_fsc(tx_o, rays_o, rays_d) #[batchsize,2]
                predict_csi = mask * predict #[batchsize,2]
                gt_csi = test_label

                predict_csi[:,0:1] = self.test_set.denormalize_rss(predict_csi[:,0:1])   #这里denorm了，就知道预测的真正结果
                gt_csi[:,0:1] = self.test_set.denormalize_rss(gt_csi[:,0:1])               
                predict_csi[:,1:2] = self.test_set.denormalize_carrier_phase(predict_csi[:,1:2])   #这里denorm了，就知道预测的真正结果
                gt_csi[:,1:2] = self.test_set.denormalize_carrier_phase(gt_csi[:,1:2])

                all_pred_csi[idx*self.batch_size:(idx+1)*self.batch_size] = predict_csi
                all_gt_csi[idx*self.batch_size:(idx+1)*self.batch_size] = gt_csi
        
        # 仅保留有效数据
        valid_indices = (all_gt_csi != 0).any(dim=1)  # 任何一列不为 0 的数据都是有效的
        valid_pred_csi = all_pred_csi[valid_indices]
        valid_gt_csi = all_gt_csi[valid_indices]        
        rss_mse, phase_mse, total_error = fsc_evaluate_2(valid_pred_csi, valid_gt_csi)

        # valid_pred_csi = valid_pred_csi[:,0:1] * torch.exp(1j * valid_pred_csi[:,1:2])
        # valid_gt_csi = valid_gt_csi[:,0:1] * torch.exp(1j * valid_gt_csi[:,1:2])

        snr = csi2snr_2(valid_pred_csi, valid_gt_csi)
        rss_acc, phase_acc = fsc_accuracy(valid_pred_csi, valid_gt_csi).unbind(dim=1)  # 分别获取 RSS 和 Carrier Phase 的准确率

        self.logger.info("Median snr:%.2f", torch.median(snr))
        self.logger.info("Median rss_acc:%.2f", torch.median(rss_acc))
        self.logger.info("Median phase_acc:%.2f", torch.median(phase_acc))
        self.logger.info("rss_mse:%.2f", rss_mse)
        self.logger.info("phase_mse:%.2f", phase_mse)
        self.logger.info("total_error:%.2f", total_error)

        scio.savemat(os.path.join(self.logdir, self.expname, "result.mat"), {'pred_csi': valid_pred_csi.cpu().numpy(),
                                                                              'gt_csi': valid_gt_csi.cpu().numpy(),
                                                                              'snr': snr.cpu().numpy(),
                                                                              'rss_acc': rss_acc.cpu().numpy(),
                                                                              'phase_acc': phase_acc.cpu().numpy(),
                                                                              'rss_mse': rss_mse.cpu().numpy(),
                                                                              'phase_mse': phase_mse.cpu().numpy(),
                                                                              'total_error': total_error.cpu().numpy(),
                                                                            })
        print("valid_pred_csi shape:", valid_pred_csi.shape)  # (4086, 2)
        print("valid_gt_csi shape:", valid_gt_csi.shape)  # (4086, 2)

        print("rss_mse shape:", rss_mse.shape)  # (4086, 1)  # 均方误差
        print("phase_mse shape:", phase_mse.shape)  # (4086, 1)
        print("total_error shape:", total_error.shape)  # (4086, 1)

        print("snr shape:", snr.shape)  # (4086, 1)  # 信噪比

        print("rss_acc shape:", rss_acc.shape)  # (4086,)  # 解绑后每列变为 1D
        print("phase_acc shape:", phase_acc.shape)  # (4086,)
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fsc-csi.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_type', type=str, default='mimo')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()

    ## backup config file
    if args.mode == 'train':
        logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
        os.makedirs(logdir, exist_ok=True)
        copyfile(args.config, os.path.join(logdir,'config.yml'))


    worker = NeRF2_Runner(mode=args.mode, dataset_type=args.dataset_type, **kwargs)
    if args.mode == 'train':
        worker.train()
    elif args.mode == 'test':
        if args.dataset_type == 'rfid':
            worker.eval_network_spectrum()
        elif args.dataset_type == 'ble':
            worker.eval_network_rssi()
        elif args.dataset_type == 'mimo':
            worker.eval_network_csi()
        elif args.dataset_type == 'fsc':
            worker.eval_network_fsc()
