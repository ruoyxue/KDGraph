import sys
import torch.nn.functional as F
import os, sys, yaml
import numpy as np
import torch
import cv2
import pickle
from torch import optim

from torch.utils.data import DataLoader

from tqdm import tqdm
from shutil import copyfile
import math
import torch.distributed

from .utils.aug_cfg import default_aug, advanced_aug, test_aug
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import build_model
from .utils.criterion import DetectorLoss

from .utils.evaluator import  APLSEvaluator
from .utils.dataset import KeypointDetectionDataset
from .utils.util import  keypoint_visualizer, vectorization_visualizer_for_graph
from .utils.patch_expansion import patch_inference


class Solver:
    def __init__(self, args, exp_path):
        self.opt = args
        with open(self.opt.config, 'r') as cfg_file:
            self.config = yaml.load(cfg_file, Loader=yaml.FullLoader)
        self.exp_path = exp_path
        
        self.dst_cfg_fn = os.path.join(self.exp_path, os.path.basename(self.opt.config))
        copyfile(self.opt.config, self.dst_cfg_fn)

        # device
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.local_rank = self.opt.local_rank
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self.use_ddp = True
                device_count = torch.cuda.device_count()
                import logging
                logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
            else:
                self.device = torch.device("cuda")
                self.use_ddp = False
                device_count = 1
        else:
            self.device = torch.device("cpu")
            self.use_ddp = False
            device_count = 1

        # dataset dataloader
        transform = None
        if self.config['DATASET']['AUGMENT'] == "default_aug":
            transform = default_aug()
        elif self.config['DATASET']['AUGMENT'] == "test_aug":
            transform = test_aug()

        train_dataset = KeypointDetectionDataset(image_path=self.config['DATASET']['TRAIN_IMAGE_PATH'],
                                                 json_path=self.config['DATASET']['TRAIN_JSON_PATH'],
                                                 pickle_path=self.config['DATASET']['GT_PICKLE_PATH'],
                                                 max_keypoint=self.config['OTHER_ARGS']['MAX_KEYPOINT'],
                                                 radius=self.config['OTHER_ARGS']['RADIUS'],
                                                 transform=transform, mode="train")
        val_dataset = KeypointDetectionDataset(image_path=self.config['DATASET']['TEST_IMAGE_PATH'],
                                               json_path=self.config['DATASET']['VAL_JSON_PATH'],
                                               pickle_path=self.config['DATASET']['GT_PICKLE_PATH'],
                                               max_keypoint=self.config['OTHER_ARGS']['MAX_KEYPOINT'] * 100,
                                               radius=self.config['OTHER_ARGS']['RADIUS'],
                                               transform=test_aug(), mode="val")
        test_dataset = KeypointDetectionDataset(image_path=self.config['DATASET']['TEST_IMAGE_PATH'],
                                                json_path=self.config['DATASET']['TEST_JSON_PATH'],
                                                pickle_path=self.config['DATASET']['GT_PICKLE_PATH'],
                                                max_keypoint=self.config['OTHER_ARGS']['MAX_KEYPOINT'] * 100,
                                                radius=self.config['OTHER_ARGS']['RADIUS'],
                                                transform=test_aug(), mode="test")

        self.num_batches_per_epoch_for_training = len(train_dataset) // self.config['TRAIN']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_validating = len(val_dataset) // self.config['VAL']['BATCH_SIZE'] // device_count
        self.num_batches_per_epoch_for_testing = len(test_dataset) // self.config['VAL']['BATCH_SIZE'] // device_count

        self.train_sampler = torch.utils.data.RandomSampler(train_dataset)
        self.val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        self.test_sampler = torch.utils.data.SequentialSampler(test_dataset)

        self.train_loader = DataLoader(dataset=train_dataset, num_workers=self.config['TRAIN']['NUM_WORKERS'], batch_size=self.config['TRAIN']['BATCH_SIZE'],
                                       sampler=self.train_sampler, drop_last=True)
        self.val_loader = DataLoader(dataset=val_dataset, num_workers=self.config['VAL']['NUM_WORKERS'], batch_size=self.config['VAL']['BATCH_SIZE'],
                                     sampler=self.val_sampler)
        self.test_loader = DataLoader(dataset=test_dataset, num_workers=self.config['VAL']['NUM_WORKERS'], batch_size=self.config['VAL']['BATCH_SIZE'],
                                      sampler=self.test_sampler)

        # model
        self.model = build_model(
            in_ch=self.config['MODEL']['IMG_CH'],
            model_key=self.config['MODEL']['NAME'],
            backbone=self.config['MODEL']['BACKBONE'],
            pretrained_flag=self.config['MODEL']['PRETRAINED_FLAG']
        ).to(self.device)

        # optimizer
        param_dicts = [
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad],
                "initial_lr": self.config['TRAIN']['OPTIMIZER']['LR']['RATE']
            }, {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.config['TRAIN']['OPTIMIZER']['LR']['BACKBONE_RATE'],
                "initial_lr": self.config['TRAIN']['OPTIMIZER']['LR']['BACKBONE_RATE']
            }
        ]
        
        if self.config['TRAIN']['OPTIMIZER']['NAME'] == 'SGD':
            self.optimizer = optim.SGD(param_dicts, self.config['TRAIN']['OPTIMIZER']['LR']['RATE'],
                                       momentum=self.config['TRAIN']['OPTIMIZER']['LR']['MOMENTUM'],
                                       weight_decay=self.config['TRAIN']['OPTIMIZER']['LR']['WEIGHT_DECAY'])
        elif self.config['TRAIN']['OPTIMIZER']['NAME'] == 'Adam':
            self.optimizer = optim.Adam(param_dicts, self.config['TRAIN']['OPTIMIZER']['LR']['RATE'],
                                        [self.config['TRAIN']['LR']['BETA1'], self.config['TRAIN']['LR']['BETA2']],
                                        weight_decay=self.config['TRAIN']['OPTIMIZER']['LR']['WEIGHT_DECAY'])
        elif self.config['TRAIN']['OPTIMIZER']['NAME'] == 'AdamW':
            self.optimizer = optim.AdamW(param_dicts, self.config['TRAIN']['OPTIMIZER']['LR']['RATE'],
                                         weight_decay=self.config['TRAIN']['OPTIMIZER']['LR']['WEIGHT_DECAY'])
        else:
            raise ValueError("wrong indicator for optimizer, which should be selected from {'SGD', 'Adam', 'AdamW'}")

        # scheduler
        if self.config['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=self.config['TRAIN']['SCHEDULER']['PATIENCE'],
                                                                       factor=self.config['TRAIN']['SCHEDULER']['FACTOR'],
                                                                       threshold=self.config['TRAIN']['SCHEDULER']['THRESHOLD'], threshold_mode='abs', verbose=True,
                                                                       min_lr=self.config['TRAIN']['SCHEDULER']['MIN_LR'])
        elif self.config['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
            milestones = self.config['TRAIN']['SCHEDULER']['MILESTONES']
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=self.config['TRAIN']['SCHEDULER']['GAMMA'], verbose=True, last_epoch=0)
        elif self.config['TRAIN']['SCHEDULER']['NAME'] == 'Poly':
            lambda1 = lambda epoch: math.pow((1 - epoch / self.config['TRAIN']['EPOCHS']) if not self.config['TRAIN']['SCHEDULER']['WARMUP'] else
                                             (1 - epoch / (self.config['TRAIN']['EPOCHS'] - int(self.config['TRAIN']['SCHEDULER']['WARMUP_EPOCH']))), 0.9)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        # loss
        self.criterion_detect = DetectorLoss(
			criterion_loc=self.config['TRAIN']['LOSS']['CRITERION_LOC'],
			criterion_dir_prob=self.config['TRAIN']['LOSS']['CRITERION_DIR_PROB'],
			criterion_dir_vec=self.config['TRAIN']['LOSS']['CRITERION_DIR_VEC'],
			lambda_loc=self.config['TRAIN']['LOSS']['LAMBDA_LOC'],
			lambda_dir_prob=self.config['TRAIN']['LOSS']['LAMBDA_DIR_PROB'],
			lambda_dir_vec=self.config['TRAIN']['LOSS']['LAMBDA_DIR_VEC']
		)
		
        # evaluator
        self.evaluator = APLSEvaluator()

        self.epochs = self.config['TRAIN']['EPOCHS']
        self.save_interval = self.config['TRAIN']['SAVE_INTERVAL']

        # whether to load checkpoint
        if self.opt.checkpoint_path == "":
            if self.use_ddp:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            self.current_epoch = 1
            self.last_acc = -999.0
            self.best_acc = -999.0
        elif os.path.exists(self.opt.checkpoint_path):
            print('checkpoint exists, loading...')
            if self.use_ddp:
                self.load_checkpoint(self.opt.checkpoint_path, rank=self.local_rank)
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            else:
                self.load_checkpoint(self.opt.checkpoint_path)
        else:
            raise FileNotFoundError(self.opt.checkpoint_path + " not exists")
    
    def load_checkpoint(self, checkpoint_path, rank=None, train_mode=True):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if rank is not None:
                if rank in [-1, 0]:
                    self.model.load_state_dict(checkpoint['net'])
                    if train_mode:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.model.load_state_dict(checkpoint['net'])
                if train_mode:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.config['TRAIN']['SCHEDULER']['NAME'] in ['MultiStepLR', "Poly"]:
                self.scheduler.last_epoch = checkpoint['epoch']
                from collections import Counter
                self.scheduler.milestones = Counter(self.config['TRAIN']['SCHEDULER']['MILESTONES'])

            self.current_epoch = checkpoint['epoch'] + 1
            self.last_acc = checkpoint['last_acc']
            self.best_acc = checkpoint['best_acc']
            self.config['TRAIN']['SCHEDULER']['WARMUP'] = False
            print('load checkpoint successfully!')
        else:
            self.current_epoch = 0
            self.last_acc = -999.0
            self.best_acc = -999.0
    
    def save_checkpoint(self, resume_path):
        state = {
            'net': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'last_acc': self.last_acc,
            'best_acc': self.best_acc
        }
        torch.save(state, resume_path)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        losses_detect = []
        losses_loc = []
        losses_dir_prob = []
        losses_dir_vec = []
        
        self.optimizer.zero_grad()
        with tqdm(total=self.num_batches_per_epoch_for_training, unit_scale=True, unit=" batch", colour="cyan", ncols=80) as pbar:
            for batch_idx, (data, targets, valid_masks) in enumerate(self.train_loader, 1):
                targets[0] = targets[0].to(self.device)
                targets[1] = targets[1].to(self.device)
                valid_masks = valid_masks.to(self.device)
                data = data.to(self.device)
                
                loc_pred, dir_pred = self.model(data)
                outputs = (loc_pred, dir_pred)
                loss_loc, loss_dir_prob, loss_dir_vec, loss_detect = \
				    self.criterion_detect(outputs, targets, valid_masks, epoch)
                losses_loc.append(loss_loc.item())
                losses_dir_prob.append(loss_dir_prob.item())
                losses_dir_vec.append(loss_dir_vec.item())
                losses_detect.append(loss_detect.item())
                loss_detect.requires_grad_(True)
                loss_detect.backward()
                if batch_idx % self.config["TRAIN"]["ACCUMULATE_BATCH_NUM"] == 0 or batch_idx == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                pbar.update()

        print("train_loss: {:.5f}".format(np.mean(losses_detect)))
        print("loss_loc: {:.5f}  loss_dir_prob: {:.5f}  loss_dir_vec: {:.5f}".format(
            np.mean(losses_loc),
            np.mean(losses_dir_prob),
            np.mean(losses_dir_vec)
        ))
        return np.mean(losses_detect)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=self.num_batches_per_epoch_for_validating, unit_scale=True, unit=" batch", colour="magenta", ncols=60) as pbar:
                for batch_idx, (data, gt_pickle_name_list) in enumerate(self.val_loader):
                    batch_size = data.shape[0]
                    gt_pickle_list = []
                    for gt_pickle_name in gt_pickle_name_list:
                        gt_pickle = pickle.load(open(gt_pickle_name, "rb"))
                        gt_pickle_list.append(gt_pickle)

                    _, pred_graphs, _, _ = patch_inference(data.to(self.device), self.model, self.config, need_keypoint=False)
                       
                    pred_pickle_list = []
                    
                    for i in range(batch_size):
                        pickle_graph = dict()
                        for node in pred_graphs[i].nodes():
                            neighbor = []
                            for n in pred_graphs[i].neighbors(node):
                                neighbor.append(n)
                            pickle_graph[node] = neighbor
                        pred_pickle_list.append(pickle_graph)
                    
                    tem_save_path = os.path.join(self.exp_path, "tem_save_path")
                    os.makedirs(tem_save_path, exist_ok=True)
                    self.evaluator.accumulate(pred_pickle_list, gt_pickle_list, tem_save_path)
                    pbar.update()

        APLS = self.evaluator.compute_metrics()
        self.evaluator.clear()
        return APLS

    def inference(self):
        self.model.eval()
        iter = tqdm(enumerate(self.test_loader), total=self.num_batches_per_epoch_for_testing, file=sys.stdout,
                    unit_scale=True, unit=" batch", colour="magenta", ncols=80)
        with torch.no_grad():
            for _, (data, image_names) in iter:
                pred_centerlines, pred_graphs, pred_locs, pred_dirs = patch_inference(
                    data.to(self.device), self.model, self.config, need_keypoint=True
                )

                for i in range(len(pred_centerlines)):
                    image = cv2.imread(os.path.join(self.config['DATASET']['TEST_IMAGE_PATH'], image_names[i]))
                    
                    # save graph as dict for apls evaluation
                    pickle_graph = dict()
                    for node in pred_graphs[i].nodes():
                        neighbor = []
                        for n in pred_graphs[i].neighbors(node):
                            neighbor.append(n)
                        pickle_graph[node] = neighbor
                    
                    with open(os.path.join(self.exp_path, "pred_graph", image_names[i].split(".")[0] + ".pickle"), "wb") as file:
                        pickle.dump(pickle_graph, file)
                    
                    vis_keypoint_output = keypoint_visualizer(image, pred_locs[i], pred_dirs[i], None)
                    vis_vec_output = vectorization_visualizer_for_graph(image, pred_graphs[i], None)
                    cv2.imwrite(os.path.join(self.exp_path, "pred_keypoint", image_names[i]), vis_keypoint_output)
                    cv2.imwrite(os.path.join(self.exp_path, "pred_vec_vis", image_names[i]), vis_vec_output)
                    cv2.imwrite(os.path.join(self.exp_path, "pred_centerline", image_names[i]), np.uint8(pred_centerlines[i] * 255))
                    
        return

    def train(self):
        if not os.path.exists(os.path.join(self.exp_path, "saved_model")):
            os.makedirs(os.path.join(self.exp_path, "saved_model"))

        start_epoch = self.current_epoch
        save_interval = self.config["TRAIN"]["SAVE_INTERVAL"]
        for epoch in range(start_epoch, self.epochs + 1):
            print('Epoch: {}'.format(self.current_epoch))
            if self.use_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            training_loss = self.train_one_epoch(epoch)

            if self.config['TRAIN']['SCHEDULER']['NAME'] == 'MultiStepLR':
                self.scheduler.step()
            if self.config['TRAIN']['SCHEDULER']['NAME'] == 'ReduceLROnPlateau':
                self.scheduler.step(metrics=training_loss)

            if epoch % save_interval == 0:
                if epoch >= self.config['TRAIN']['EPOCH_TO_START_VALID']:
                    if self.config["TRAIN"]["ONLY_SAVE_BEST_MODEL"]:
                        # only save last and best model
                        self.save_checkpoint(os.path.join(self.exp_path, "saved_model", "last_model.pt"))
                        APLS = self.validate()
                        print(f"APLS: {APLS}")
                        self.last_acc = APLS
                        with open(os.path.join(self.exp_path, "saved_model", "last_model.txt"), 'w') as f:
                            f.write(f"Last model epoch {self.current_epoch}:\n")
                            f.write(f"APLS: {APLS}")

                        if APLS > self.best_acc:
                            self.best_acc = APLS
                            self.save_checkpoint(os.path.join(self.exp_path, "saved_model", "best_model.pt"))
                            with open(os.path.join(self.exp_path, "saved_model", "best_model.txt"), 'w') as f:
                                f.write(f"Best model epoch {self.current_epoch}:\n")
                                f.write(f"APLS: {APLS}")
                            print(
                                "----------------best model saved-----------------\n")
                        else:
                            print("best APLS: {:.4f}".format(self.best_acc))
                            print("\n")

                    else:
                        # save models every epoch interval
                        current_model_fn = os.path.join(self.exp_path, "saved_model", f"epoch_{self.current_epoch}_model.pt")
                        self.save_checkpoint(current_model_fn)
                        APLS = self.validate()
                        print(f"APLS: {APLS}")
                        with open(os.path.join(self.exp_path, "saved_model", "log.txt"), 'a') as f:
                            f.write(f"epoch {self.current_epoch}:\n")
                            f.write(f"APLS: {APLS}\n")
                else:
                    current_model_fn = os.path.join(self.exp_path, "saved_model", "last_model.pt")
                    self.save_checkpoint(current_model_fn)
                    with open(os.path.join(self.exp_path, "saved_model", "last_model.txt"), 'w') as f:
                        f.write(f"Last model epoch: {self.current_epoch}:\n")

            self.current_epoch += 1

