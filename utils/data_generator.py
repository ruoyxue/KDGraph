import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import math


class GTGenerator:
    """ Used in keypoint matching network """
    @staticmethod
    def generate(output_shape, location, direction, radius=1):
        """ Generate location and direction gt feature maps

        Notes:
            output_shape: (int, int)
            location: (max_keypoint, 2)
            direction: (max_keypoint, 6, 3)

        Args:
            radius: radius of Gaussian kernel in location maps

        """
        gt_location_map, gt_direction_map = \
            GTGenerator.get_map(location, direction, radius, output_shape)
            
        return gt_location_map, gt_direction_map

    @staticmethod
    def get_direction_map_gaussian(gt_direction_map, location, direction, radius):
        num_keypoints = location.shape[0]
        diameter = 2 * radius + 1
        gaussian = GTGenerator.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)

        for i in range(num_keypoints):
            x, y = int(location[i][0]), int(location[i][1])
            if x >= 0 and y >= 0:
                height, width = gt_direction_map.shape[-2:]

                left, right = min(x, radius), min(width - x, radius + 1)
                top, bottom = min(y, radius), min(height - y, radius + 1)

                masked_fmap = gt_direction_map[:, y - top:y + bottom, x - left:x + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
                    # TODO: optimize code here
                    for m in range(masked_fmap.shape[1]):
                        for n in range(masked_fmap.shape[2]):
                            if masked_fmap[0, m, n] < masked_gaussian[m, n]:
                                masked_fmap[:, m, n] = torch.Tensor(direction[i].flatten())
                                masked_fmap[[0, 3, 6, 9, 12, 15], m, n] = masked_gaussian[m, n]

                    gt_direction_map[:, y - top:y + bottom, x - left:x + right] = masked_fmap

    @staticmethod
    def get_map(location, direction, radius, output_shape):
        output_height, output_width = output_shape
        gt_direction_map = torch.zeros(output_height, output_width, 18)
        gt_location_map = torch.zeros(output_height, output_width)

        num_keypoints = location.shape[0]
        diameter = 2 * radius + 1
        gaussian = GTGenerator.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)

        for i in range(num_keypoints):
            x, y = int(location[i][0]), int(location[i][1])
            if x >= 0 and y >= 0:
                left, right = min(x, radius), min(output_width - x, radius + 1)
                top, bottom = min(y, radius), min(output_height - y, radius + 1)

                masked_dir = gt_direction_map[y - top:y + bottom, x - left:x + right, :]
                masked_loc = gt_location_map[y - top:y + bottom, x - left:x + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

                if min(masked_gaussian.shape) > 0 and min(masked_loc.shape) > 0:
                    select_matrix = (masked_loc < masked_gaussian)  # where to assign new direction
                    masked_loc = torch.max(masked_loc, masked_gaussian)
                    masked_dir[select_matrix] = torch.Tensor(direction[i].flatten())
                    gt_direction_map[y - top:y + bottom, x - left:x + right, :] = masked_dir
                    gt_location_map[y - top:y + bottom, x - left:x + right] = masked_loc

        gt_direction_map = gt_direction_map.permute(2, 0, 1)
        gt_location_map = gt_location_map.unsqueeze(0)
        return gt_location_map, gt_direction_map

    @staticmethod
    def get_location_map(gt_location_map, location, radius, k=1):
        num_keypoints = location.shape[0]
        diameter = 2 * radius + 1
        gaussian = GTGenerator.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)

        for i in range(num_keypoints):
            x, y = int(location[i][0]), int(location[i][1])
            if x >= 0 and y >= 0:
                height, width = gt_location_map.shape[-2:]

                left, right = min(x, radius), min(width - x, radius + 1)
                top, bottom = min(y, radius), min(height - y, radius + 1)

                masked_fmap = gt_location_map[0, y - top:y + bottom, x - left:x + right]
                masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
                    masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
                    gt_location_map[0, y - top:y + bottom, x - left:x + right] = masked_fmap

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def get_mask(gt_mask, location):
        """ mask used to indicate where there is a keypoint in feature map, is used in loss calculation"""
        num_keypoints = location.shape[0]
        for i in range(num_keypoints):
            x, y = int(location[i][0]), int(location[i][1])
            if x >= 0 and y >= 0:
                gt_mask[0, y, x] = 1


def sort_direction(direction, max_direction=6):
    """
    Sort direction to different angle partitions, such as 0-60, 6-120 etc,
    every angle partition can have one direction at most.

    Notes:
        direction (list): [sin, cos, sin, cos...]
        max_direction: number of angle partitions
    """
    def get_degree(sin_value, cos_value):
        """ get angle in [0, 360) via sin and cos value """
        theta = math.atan2(sin_value, cos_value)
        if theta < 0:
            theta += 2 * math.pi
        return math.degrees(theta)

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


def update_direction(loc_idx, direction_list, direction):
    """
    Sort direction after flip or rotate using 'sort_direction' and update direction

    Notes:
        direction_list: [sin, cos, sin, cos...]
        direction: (max_keypoint, 6, 3)

    Args:
        loc_idx: location index of current keypoint
        direction_list: direction to be sorted
        direction: output direction array
    """
    max_direction = direction.shape[1]
    sorted_direction_list = sort_direction(direction_list, max_direction=max_direction)
    direction[loc_idx] = 0  # set to zero
    for i in range(max_direction):
        if sorted_direction_list[2 * i] is not None:
            direction[loc_idx, i, 0] = 1
            direction[loc_idx, i, 1] = sorted_direction_list[2 * i]
            direction[loc_idx, i, 2] = sorted_direction_list[2 * i + 1]
    return direction

