import copy
import os, yaml
import torch
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.aug_cfg import test_aug
from model import build_model
from utils.dataset import KeypointDetectionDataset
from utils.feature_vis import KDGraph_Feature_Visualizer


class VisSolver:
    def __init__(self, args, exp_path):
        self.opt = args
        with open(self.opt.config, 'r') as cfg_file:
            self.config = yaml.load(cfg_file, Loader=yaml.FullLoader)
        self.exp_path = exp_path
        self.device = torch.device("cuda")
        device_count = 1
        self.use_ddp = False

        test_dataset = KeypointDetectionDataset(image_path=self.config['DATASET']['TEST_IMAGE_PATH'],
                                                json_path=self.config['DATASET']['TEST_JSON_PATH'],
                                                pickle_path=self.config['DATASET']['GT_PICKLE_PATH'],
                                                max_keypoint=self.config['OTHER_ARGS']['MAX_KEYPOINT'],
                                                radius=self.config['OTHER_ARGS']['RADIUS'],
                                                transform=test_aug(), mode="test")

        self.num_batches_per_epoch_for_testing = len(test_dataset) // device_count
        self.test_sampler = torch.utils.data.SequentialSampler(test_dataset)

        self.test_loader = DataLoader(dataset=test_dataset, num_workers=self.config['VAL']['NUM_WORKERS'],
                                      batch_size=1, sampler=self.test_sampler)

        # model
        self.model = build_model(
            in_ch=self.config['MODEL']['IMG_CH'],
            model_key=self.config['MODEL']['NAME'],
            backbone=self.config['MODEL']['BACKBONE'],
            pretrained_flag=self.config['MODEL']['PRETRAINED_FLAG']
        ).to(self.device)

        # whether to load checkpoint
        if os.path.exists(self.opt.checkpoint_path):
            print('checkpoint exists, loading...')
            self.load_checkpoint(self.opt.checkpoint_path)
        else:
            raise FileNotFoundError(self.opt.checkpoint_path + " not exists")

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['net'])

    def feature_visualise(self):
        self.model.eval()
        
        if self.config['MODEL']['NAME'] == "KDGraph":
            visualizer = KDGraph_Feature_Visualizer(self.model)
       
        with torch.no_grad():
            with tqdm(total=self.num_batches_per_epoch_for_testing, unit_scale=True, unit=" batch", colour="magenta", ncols=80) as pbar:
                for _, (data, image_names) in enumerate(self.test_loader):
                    data = data.to(self.device)
                    image = cv2.imread(os.path.join(self.config['DATASET']['TEST_IMAGE_PATH'], image_names[0]))
                    crop_size = self.config["VAL"]["CROP_SIZE"]
                    overlap_size = self.config["VAL"]["OVERLAP_SIZE"]
                    batch_size, _, height, width = data.shape
                    if crop_size < height or crop_size < width:
                        # small patch vis
                        x_list = list(range(0, width - crop_size, crop_size - overlap_size)) + [width - crop_size]
                        y_list = list(range(0, height - crop_size, crop_size - overlap_size)) + [height - crop_size]
                        left_upper_coord_list = [(x, y) for y in y_list for x in x_list]

                        patch_count = 0
                        for x, y in left_upper_coord_list:
                            data_patch = data[:, :, y: y + crop_size, x: x + crop_size]

                            target_output_list = visualizer.get_target_output(data_patch)
                            patch_image = copy.deepcopy(image[y: y + crop_size, x: x + crop_size, :])
                            visualizer.save_feature(image=patch_image, target_output_dict=target_output_list,
                                                    save_dir=os.path.join(self.exp_path),
                                                    apply_feat_on_image=True)
                            patch_count += 1

                    else:
                        # whole image vis
                        target_output_list = visualizer.get_target_output(data)
                        visualizer.save_feature(image=image, target_output_dict=target_output_list,
                                                save_dir=os.path.join(self.exp_path), image_name=image_names[0],
                                                apply_feat_on_image=True)

                    pbar.update()
