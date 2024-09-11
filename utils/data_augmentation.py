import random
from typing import Tuple, List
import math
import cv2
import numpy as np
import skimage
import torch
import imutils
from scipy.optimize import linear_sum_assignment
import copy


class ProcessingSequential:
    """ Sequential of preprocessing methods """
    def __init__(self, sequence: List):
        self.sequence = sequence

    def __call__(self, image, location, direction, valid_mask):
        for processing in self.sequence:
            image, location, direction, valid_mask = processing(image, location, direction, valid_mask)
        return image, location, direction, valid_mask

    def __getitem__(self, item: int):
        if not isinstance(item, int):
            raise RuntimeError("data_aug ProcessingSequential needs int as index, not string")
        return self.sequence[item]

    def __repr__(self):
        output_string = ""
        for processing in self.sequence:
            output_string += processing.__class__.__name__ + "("
            for key in processing.__dict__.keys():
                output_string += key + "=" + str(processing.__dict__[key]) + ", "
            output_string = output_string.strip(", ") + ")\n"
        return output_string.strip("\n")

    def list_of_repr(self):
        """ store str of each preprocessing method in a list and return """
        output_list = []
        for processing in self.sequence:
            pro_str = processing.__class__.__name__ + "("
            for key in processing.__dict__.keys():
                pro_str += key + "=" + str(processing.__dict__[key]) + ", "
            pro_str = pro_str.strip(", ") + ")"
            output_list.append(pro_str)
        return output_list

def get_degree(sin_value, cos_value):
    """ get angle in [0, 360) via sin and cos value """
    theta = math.atan2(sin_value, cos_value)
    if theta < 0:
        theta += 2 * math.pi
    return math.degrees(theta)

def sort_direction(direction, max_direction=6):
    """
    Sort direction to different angle partitions, such as 0-60, 6-120 etc,
    every angle partition can have one direction at most.

    Notes:
        direction (list): [sin, cos, sin, cos...]
        max_direction: number of angle partitions
    """

    assert len(direction) % 2 == 0
    num_direction = int(len(direction) / 2)
    angles = []
    partition_middles = []

    for tem in range(max_direction):
        partition_middles.append((2 * tem + 1) * 360 / (2 * max_direction))

    for tem in range(num_direction):
        angles.append(get_degree(direction[2 * tem], direction[2 * tem + 1]))

    angles = np.array(angles)
    angles = angles[:, np.newaxis]
    partition_middles = np.array(partition_middles)
    partition_middles = partition_middles[np.newaxis, :]
    cost_matrix = abs(angles - partition_middles)
    row, col = linear_sum_assignment(cost_matrix)

    output_direction = []
    for tem in range(max_direction):
        if tem in col:
            index = list(col).index(tem)
            output_direction.append(direction[2 * row[index]])
            output_direction.append(direction[2 * row[index] + 1])
        else:
            output_direction.append(None)
            output_direction.append(None)
    return output_direction

def sort_direction_in_order(direction, max_direction):
    """
    Sort direction based on their angles (ascending order)
    Notes:
        direction (list): [sin, cos, sin, cos...]
        max_direction: number of angle partitions
    """
    assert len(direction) % 2 == 0
    num_direction = int(len(direction) / 2)
    angles = []
    output_direction = []

    for tem in range(min(num_direction, max_direction)):
        angles.append(get_degree(direction[2 * tem], direction[2 * tem + 1]))

    sorted_angles = copy.deepcopy(angles)
    sorted_angles.sort()

    for angle in sorted_angles:
        idx = angles.index(angle)
        if len(output_direction) < 2 * max_direction:
            output_direction.append(direction[2 * idx])
            output_direction.append(direction[2 * idx + 1])
    
    if len(output_direction) < 2 * max_direction:
        for _ in range(max_direction - num_direction):
            output_direction.append(None)
            output_direction.append(None)

    return output_direction

def sort_direction_without_order(direction, max_direction):
    """
    Format direction without order
    Notes:
        direction (list): [sin, cos, sin, cos...]
        max_direction: number of angle partitions
    """
    assert len(direction) % 2 == 0
    num_direction = int(len(direction) / 2)
    angles = []
    output_direction = []

    for tem in range(min(num_direction, max_direction)):
        output_direction.append(direction[2 * tem])
        output_direction.append(direction[2 * tem + 1])

    if len(output_direction) < 2 * max_direction:
        for _ in range(max_direction - num_direction):
            output_direction.append(None)
            output_direction.append(None)

    return output_direction

def update_direction(loc_idx, direction_list, direction, mode):
    """
    Sort direction after flip or rotate using 'sort_direction' and update direction

    Notes:
        direction_list: [sin, cos, sin, cos...]
        direction: (max_keypoint, max_direction, 3)

    Args:
        loc_idx: location index of current keypoint
        direction_list: direction to be sorted
        direction: output direction array
        mode: allocate, order, without_order
    """
    max_direction = direction.shape[1]
    if mode == 'allocate':
        sorted_direction_list = sort_direction(direction_list, max_direction=max_direction)
    elif mode == 'order':
        sorted_direction_list = sort_direction_in_order(direction_list, max_direction=max_direction)
    elif mode == 'without_order':
        sorted_direction_list = sort_direction_without_order(direction_list, max_direction=max_direction)
    direction[loc_idx] = 0  # set to zero
    for i in range(max_direction):
        if sorted_direction_list[2 * i] is not None:
            direction[loc_idx, i, 0] = 1
            direction[loc_idx, i, 1] = sorted_direction_list[2 * i]
            direction[loc_idx, i, 2] = sorted_direction_list[2 * i + 1]
    return direction


class RandomFlip(object):
    """
    Randomly  flips the Image horizontally or vertically

    Args:
        p_horizontal: the probability with which the image is flipped horizontally
        p_vertical: the probability with which the image is flipped vertically
    """
    def __init__(self, p_horizontal=0.5, p_vertical=0.5, mode="allocate"):
        self.p_horizontal = float(p_horizontal)
        self.p_vertical = float(p_vertical)
        self.mode = mode

    def __call__(self, image, location, direction, valid_mask):
        """
        Notes:
            image: (height, width, 3)
            location: (max_keypoint, 2)
            direction: (max_keypoint, 6, 3)
            valid_mask: (height, width)
        """
        height, width, _ = image.shape
        num_valid_keypoint = torch.all(location >= 0, dim=1).sum()

        if random.random() < self.p_horizontal:
            # horizontally flip
            image = image[:, ::-1, :]
            valid_mask = valid_mask[:, ::-1]
            for loc_idx in range(num_valid_keypoint):
                location[loc_idx, 0] = width - location[loc_idx, 0] - 1  # adjust x
                direction_list = []
                for dir_idx in range(6):
                    if direction[loc_idx, dir_idx, 0] != 0:
                        direction[loc_idx, dir_idx, 2] *= -1  # adjust cos
                        direction_list.append(direction[loc_idx, dir_idx, 1].item())
                        direction_list.append(direction[loc_idx, dir_idx, 2].item())
                direction = update_direction(loc_idx, direction_list, direction, self.mode)

        if random.random() < self.p_vertical:
            # vertically flip
            image = image[::-1, :, :]
            valid_mask = valid_mask[::-1, :]
            for loc_idx in range(num_valid_keypoint):
                location[loc_idx, 1] = height - location[loc_idx, 1] - 1  # adjust y
                direction_list = []
                for dir_idx in range(6):
                    if direction[loc_idx, dir_idx, 0] != 0:
                        direction[loc_idx, dir_idx, 1] *= -1  # adjust sin
                        direction_list.append(direction[loc_idx, dir_idx, 1].item())
                        direction_list.append(direction[loc_idx, dir_idx, 2].item())
                direction = update_direction(loc_idx, direction_list, direction, self.mode)

        image = image.copy()
        valid_mask = valid_mask.copy()
        return image, location, direction, valid_mask


class RandomRotate(object):
    def __init__(self, mode='allocate'):
        self.mode = mode

    """ Randomly rotates an image, choice in [0, 90, 180, 270] """
    def __call__(self, image, location, direction, valid_mask):
        """
        Notes:
            image: (height, width, 3)
            location: (max_keypoint, 2)
            direction: (max_keypoint, 6, 3)
            valid_mask (height, width)

        """
        angle = random.choice([0, 90, 180, 270])
        height, width, _ = image.shape
        image = imutils.rotate_bound(image, angle)
        valid_mask = imutils.rotate_bound(valid_mask, angle)
        num_valid_keypoint = torch.all(location >= 0, dim=1).sum()

        for loc_idx in range(num_valid_keypoint):
            # rotate locations
            if angle == 0:
                continue
            x, y = (location[loc_idx, 0]).float(), location[loc_idx, 1].float()
            center_x, center_y = (width - 1) / 2, (height - 1) / 2
            y = height - y - 1
            center_y = height - center_y - 1
            new_x = (x - center_x) * math.cos(math.radians(angle)) - (y - center_y) * math.sin(math.radians(angle)) + center_x
            new_y = (x - center_x) * math.sin(math.radians(angle)) + (y - center_y) * math.cos(math.radians(angle)) + center_y
            if angle == 90 or angle == 270:
                location[loc_idx, 0] = torch.round(width - new_x - 1)
                location[loc_idx, 1] = torch.round(new_y)
            elif angle == 180:
                location[loc_idx, 0] = torch.round(new_x)
                location[loc_idx, 1] = torch.round(height - new_y - 1)

            direction_list = []
            for dir_idx in range(6):
                if direction[loc_idx, dir_idx, 0] != 0:
                    # rotate angles
                    dir_angle = torch.atan2(direction[loc_idx, dir_idx, 1], direction[loc_idx, dir_idx, 2])
                    dir_rad_after_rotate = torch.deg2rad(torch.rad2deg(dir_angle) + angle)
                    direction[loc_idx, dir_idx, 1] = torch.sin(dir_rad_after_rotate)
                    direction[loc_idx, dir_idx, 2] = torch.cos(dir_rad_after_rotate)
                    direction_list.append(direction[loc_idx, dir_idx, 1].item())
                    direction_list.append(direction[loc_idx, dir_idx, 2].item())
            direction = update_direction(loc_idx, direction_list, direction, self.mode)

        return image, location, direction, valid_mask


class Normalize:
    """ mean and std need order (B, G, R) """
    def __init__(self, mean: Tuple, std: Tuple, mode="mean_std"):
        assert len(mean) == len(std), \
            f"Preprocessing Normalise expects same-size mean and std, got {len(mean)} and {len(std)}"
        self.mean = mean
        self.std = std
        assert mode in ["mean_std", "simple"]
        self.mode = mode

    def __call__(self, image, locations, directions, valid_mask):
        """
        Notes:
            image: (height, width, 3)  numpy.array
            location: (max_keypoint, 2)
            direction: (max_keypoint, 6, 3)
            valid_mask: (height, width)  numpy.array

        Returns:
            image: (3, height, width)  torch.Tensor
            segmentation(1, height, height)  torch.Tensor
        """
        assert len(self.mean) == image.shape[2], \
            f"Preprocessing Normalise len(mean) must equal to ({len(self.mean)}) and image.shape[2]({image.shape[2]})"
        image = torch.Tensor(image)
        valid_mask = torch.Tensor(valid_mask)
        for i in range(len(self.mean)):
            if self.mode == "mean_std":
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
            elif self.mode == "simple":
                image[:, :, i] = (image[:, :, i] / 255) * 2 - 1  # (-1, 1)
        image = image.permute(2, 0, 1)
        valid_mask = valid_mask.unsqueeze(0)
        return image, locations, directions, valid_mask


class AddNoise:
    """ add noise to image using skimage.util.random_noise """
    def __init__(self, mode, var, p=0.5):
        self.mode = mode
        self.var = var
        self.p = p

    def __call__(self, image: np.array, location: torch.tensor, direction: torch.tensor, valid_mask):
        if random.random() < self.p:
             image = np.uint8(skimage.util.random_noise(image, self.mode, var=self.var) * 256)
        return image, location, direction, valid_mask


class RandomVariation:
    """ image variation
    :param mode: random_brightness, random_hue, random_saturation, random_contrast
    """
    def __init__(self, mode, p=0.1, factor=[0.9, 1.1]):
        assert mode in ["random_brightness", "random_hue", "random_saturation", "random_contrast"],\
            f"Augmentation RandomVariation expects mode in [random_brightness, " \
            f"random_hue, random_saturation, random_contrast], got {mode}"
        self.mode = mode
        self.p = p
        self.factor = factor

    def __call__(self, image: np.array, location: torch.tensor, direction: torch.tensor, valid_mask):
        if np.random.random() < self.p:
            random_value = np.random.random() * (self.factor[1] - self.factor[0]) + self.factor[0]

            if self.mode == "random_brightness":
                H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                V = V.astype(float)
                V = np.clip(V * random_value, a_min=0, a_max=255).astype(np.uint8)
                image = cv2.cvtColor(cv2.merge([H, S, V]), cv2.COLOR_HSV2BGR)
            elif self.mode == "random_saturation":
                H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                S = S.astype(float)
                S = np.clip(S * random_value, a_min=0, a_max=255).astype(np.uint8)
                image = cv2.cvtColor(cv2.merge([H, S, V]), cv2.COLOR_HSV2BGR)
            elif self.mode == "random_hue":
                H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                H = H.astype(float)
                H = np.clip(H * random_value, a_min=0, a_max=180).astype(np.uint8)
                image = cv2.cvtColor(cv2.merge([H, S, V]), cv2.COLOR_HSV2BGR)

        return image, location, direction, valid_mask