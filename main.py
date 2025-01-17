import argparse
import os
import time

import torch
import torch.distributed as dist
import yaml
from solver import Solver
from vis_solver import VisSolver

def main(opt):
    with open(opt.config, 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)
        ROOT_PATH = config["ROOT_PATH"]
    # -------------- pretrain detector ---------------
    if opt.mode == 'train':
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        exp_path = os.path.join(ROOT_PATH, 'pretrain_detector', f"{time_stamp} ({opt.log})")
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            torch.cuda.set_device(opt.local_rank)
            dist.init_process_group(backend='nccl')
            if dist.get_rank() in [-1, 0]:
                os.makedirs(exp_path)
                dist.barrier()
            else:
                dist.barrier()
        else:
            os.makedirs(exp_path)
        
        solver = Solver(opt, exp_path)
        solver.train()
    elif opt.mode == 'test':
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            raise ValueError("test mode does not support ddp!")
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        exp_path = os.path.join(ROOT_PATH, 'test_detector', f"{time_stamp}({opt.log})")
        os.makedirs(exp_path)
        os.makedirs(os.path.join(exp_path, "pred_keypoint"))
        os.makedirs(os.path.join(exp_path, "pred_vec_vis"))
        os.makedirs(os.path.join(exp_path, "pred_centerline"))
        os.makedirs(os.path.join(exp_path, "pred_graph"))
        solver = Solver(opt, exp_path)
        solver.inference()
    elif opt.mode == 'vis':
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            raise ValueError("vis mode does not support ddp!")
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        exp_path = os.path.join(ROOT_PATH, 'feature_vis', f"{time_stamp} ({opt.log})")
        os.makedirs(exp_path)
        vis_solver = VisSolver(opt, exp_path)
        vis_solver.feature_visualise()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='the config file used to set training process')
    parser.add_argument('--checkpoint_path', type=str, default='', help='the checkpoint file for loading')
    parser.add_argument('--mode', type=str, default='train', help='mode of process')
    parser.add_argument('--log', type=str, default='', help='add log info to save directory')

    opt = parser.parse_args()
    main(opt)
