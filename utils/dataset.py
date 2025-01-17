import json
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from .data_generator import GTGenerator


class KeypointDetectionDataset(Dataset):
    """ Used for pretraining of keypoint detection network """
    def __init__(self, image_path, json_path, pickle_path, max_keypoint=350, radius=3,
                 transform=None, mode="train"):
        self.image_path = image_path
        self.json_path = json_path
        self.pickle_path = pickle_path
        self.transform = transform
        self.max_keypoint = max_keypoint
        self.mode = mode
        self.radius = radius
        with open(json_path, 'r', encoding='utf-8') as f:
            json_info = json.load(f)
            self.image_info_list = json_info["dataset_info"]
        self.pickle_list = os.listdir(pickle_path)
        self.image_list = []

    def __len__(self):
        return len(self.image_info_list)

    def __getitem__(self, idx):
        location, direction, image_shape, image_name = self.get_info_from_idx(idx)

        image = cv2.imread(os.path.join(self.image_path, image_name))

        ignore_mask = (image[:, :, 0] == 5) * (image[:, :, 1] == 5) * (image[:, :, 2] == 5)
        valid_mask = (ignore_mask == 0).astype(float)
        height, width = image_shape[:2]

        if self.transform is not None:
            image, location, direction, valid_mask = self.transform(image, location, direction, valid_mask)

        if self.mode == "train":
            location = self.random_shift_margin_location(location, radius=10, output_shape=(height, width))

            gt_location_map, gt_direction_map = \
                GTGenerator.generate((height, width), location, direction, radius=self.radius)

            return image, (gt_location_map, gt_direction_map), valid_mask
       
        if self.mode == "val":
            pickle_name = image_name.split(".")[0] + ".pickle"
            return image, os.path.join(self.pickle_path, pickle_name), image_name, (height, width)
        
        if self.mode == "test":
            return image, image_name

    def get_info_from_idx(self, idx):
        image_info = self.image_info_list[idx]
        image_name = image_info["image_name"]
        height = image_info["height"]
        width = image_info["width"]
        location = torch.full((self.max_keypoint, 2), -1)
        direction = torch.zeros(self.max_keypoint, 6, 3)
        keypoints = image_info["keypoints"]

        count = 0
        for keypoint in keypoints:
            location[count, :] = torch.Tensor(keypoint["location"])
            dir_count = 0
            keypoint_dirs = keypoint["directions"]
            for i in range(0, len(keypoint_dirs), 2):
                if (keypoint_dirs[i] is not None) and (keypoint_dirs[i + 1] is not None):
                    direction[count, dir_count, 0] = 1
                    direction[count, dir_count, 1:] = torch.Tensor(keypoint_dirs[i: i+2])
                dir_count += 1
            count += 1

        return location, direction, (height, width), image_name

    def random_shift_margin_location(self, location, radius=5, output_shape=(1024, 1024)):
        """
        Random move keypoint location in a small range without changing direction and adjacency matrix

        Notes:
            location: (max_keypoint, 2) torch.Tensor

        Args:
            p: probability for a keypoint to be moved
            radius: radius of range
        """
        num_valid_keypoint = torch.all(location >= 0, dim=1).sum()
        for i in range(num_valid_keypoint):
            if location[i, 0] == 0:
                move_x = np.random.randint(low=1, high=radius + 1)
                if 0 <= location[i, 0] + move_x < output_shape[1]:
                        location[i, 0] += move_x
            if location[i, 0] == output_shape[1] - 1:
                move_x = np.random.randint(low=-radius, high=0)
                if 0 <= location[i, 0] + move_x < output_shape[1]:
                    location[i, 0] += move_x
            if location[i, 1] == 0:
                move_y = np.random.randint(low=1, high=radius + 1)
                if 0 <= location[i, 1] + move_y < output_shape[0]:
                    location[i, 1] += move_y
            if location[i, 1] == output_shape[0] - 1:
                move_y = np.random.randint(low=-radius, high=0)
                if 0 <= location[i, 1] + move_y < output_shape[0]:
                    location[i, 1] += move_y
        return location
