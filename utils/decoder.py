import torch
import torch.nn.functional as F
import copy
import numpy as np
import cv2
from skimage import morphology
import math
from scipy.optimize import linear_sum_assignment


class Decoder:
    """ used to transform features to detection results """
    @staticmethod
    def keypoint_decode(loc, direction, max_keypoint=350, nms_poolsize=3, loc_threshold=0.5, dir_threshold=0.5):
        """
        Decode output feature map to keypoint results

        Notes:
            loc: (batch, 1, height, width)
            direction: (batch, 18, height, width)

        Args:
            max_keypoint: select top k features in loc
            loc_threshold: threshold for determining low confidence location
            dir_threshold: threshold for determining low direction location
        """
        batch, _, height, width = loc.shape
        loc = loc.cpu()
        direction = direction.cpu()

        loc = KeyPointDecoder.pseudo_nms(loc, pool_size=nms_poolsize)
        scores, index, ys, xs = KeyPointDecoder.topk_score(loc, max_keypoint)
        location = torch.zeros(batch, max_keypoint, 2)

        # remove low confidence keypoints
        row, col = torch.where(scores < loc_threshold)
        ys[[row, col]], xs[[row, col]] = -1, -1
        location[:, :, 0] = xs
        location[:, :, 1] = ys

        direction = KeyPointDecoder.gather_feature(direction, index, use_transform=True)
        direction = direction.reshape(batch, max_keypoint, 6, 3)

        # normalise direction vector
        length = torch.clamp(torch.sqrt(direction[:, :, :, 1] ** 2 + direction[:, :, :, 2] ** 2), 1e-8)
        direction[:, :, :, 1] /= length
        direction[:, :, :, 2] /= length

        # remove low confidence direction
        direction[direction[:, :, :, 0] < dir_threshold] = 0

        return location.int(), direction  # location (batch, max_keypoint, 2), direction (batch, max_keypoint, 6, 3)

    @staticmethod
    def keypoint_decode_skeleton(loc, direction, loc_threshold=0.5, dir_threshold=0.5, keypoint_pixel_threshold=10,
                                 use_sigmoid=False):
        """
        Decode output feature map to keypoint results

        Notes:
            loc: (1, height, width)
            direction: (18, height, width)

        Args:
            loc_threshold: threshold for loc binarization
            dir_threshold: threshold for low confidence direction removal
            keypoint_pixel_threshold: threshold to determine whether to take pixels as
                                      isolated keypoint or road segment
            use_sigmoid: whether to use sigmoid for loc and direction prob
        """
        _, height, width = loc.shape
        loc = loc.cpu()
        direction = direction.cpu()
        loc_road_segment = np.zeros((loc.shape[1], loc.shape[2]))  # record road segments, without keypoints
        locations = []
        directions = []

        if use_sigmoid:
            loc = loc - torch.min(loc) / torch.clamp(torch.max(loc) - torch.min(loc), min=1e-8)
            loc = 4 * loc - 2
            loc = torch.sigmoid(loc)
            direction[[0, 3, 6, 9, 12, 15], :, :] = torch.sigmoid(direction[[0, 3, 6, 9, 12, 15], :, :])

        loc = loc.squeeze().numpy()  # loc (height, width)
        direction = direction.permute(1, 2, 0).numpy()  # direction (height, width, 18)
        _, loc_binary = cv2.threshold(loc, loc_threshold, 1, cv2.THRESH_BINARY)

        # generate loc skeleton without small holes
        loc_skeleton = morphology.skeletonize(loc_binary > 0).astype(np.uint8)
        # cv2.imshow("ske_before", loc_skeleton * 255)
        loc_skeleton = cv2.dilate(loc_skeleton, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # cv2.imshow("ske_dilation", loc_skeleton * 255)
        contours, _ = cv2.findContours(loc_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                cv_contours.append(contour)
        cv2.fillPoly(loc_skeleton, cv_contours, 1)
        loc_skeleton = morphology.skeletonize(loc_skeleton > 0).astype(np.uint8)
        # cv2.imshow("ske_after", loc_skeleton * 255)

        num_points, labels_map = cv2.connectedComponents(loc_skeleton, connectivity=8)

        for keypoint_label in range(1, num_points):
            (row, col) = np.where(labels_map == keypoint_label)
            keypoint_pixel_sum = len(row)
            if keypoint_pixel_sum < keypoint_pixel_threshold:
                # should be viewed as isolated keypoint
                # take pixel with max loc feature as keypoint
                keypoint_loc_value_list = loc[[row, col]]
                max_value_index = np.argmax(keypoint_loc_value_list)
                y, x = row[max_value_index], col[max_value_index]
                locations.append([x, y])
                directions.append(direction[y, x])
            else:
                # should be viewed as road segment
                # find endpoints in these pixels as keypoints
                keypoint_coordinates_list = [[y, x] for y, x in zip(row, col)]
                for (y, x) in keypoint_coordinates_list:
                    loc_road_segment[y, x] = 1
                endpoints = KeyPointDecoder.find_endpoints(loc_skeleton, keypoint_coordinates_list)
                for (y, x) in endpoints:
                    locations.append([x, y])
                    directions.append(direction[y, x])

        locations = np.array(locations)
        directions = np.reshape(np.array(directions), (-1, 6, 3))

        # normalise direction vector
        length = np.clip(np.sqrt(direction[:, :, 1] ** 2 + direction[:, :, 2] ** 2), a_min=1e-8, a_max=1e8)
        direction[:, :, 1] /= length
        direction[:, :, 2] /= length

        # remove low confidence direction
        direction[direction[:, :, 0] < dir_threshold] = 0

        # location (num_valid_keypoint, 2)
        # loc_road_segment (height, width)
        # direction (num_valid_keypoint, 6, 3)
        return locations, loc_road_segment, directions

    @staticmethod
    def vectorization_decode(location, direction, distance_range=[30, 448], point_line_distance=[50, 140]):
        """
        Decode locations and directions to adjacency matrix

        Notes:
            location: (batch, max_keypoint, 2)
            direction: (batch, max_keypoint, 6, 3)

        Args:
            distance_range: distance range for criterion adjustment
            point_line_distance: distance range between one keypoint and other matched keypoint's direction

        Returns:
            adjacency matrix of locations
        """
        assert location.shape[:2] == direction.shape[:2]
        locations = location.float()
        directions = copy.deepcopy(direction)
        batch_size, max_keypoint = location.shape[:2]

        adjacency_matrix = torch.zeros(batch_size, max_keypoint, max_keypoint)
        for batch in range(batch_size):
            adjacency_matrix[batch, :, :] = VectorizationDecoder.calculate_adjacency_matrix_closest(
                location=locations[batch], direction=directions[batch],
                distance_range=distance_range, point_line_distance=point_line_distance
            )

        return adjacency_matrix.int()

    @staticmethod
    def vectorization_decode_hungarian(location, direction, lambda_angle_dist, lambda_point_line_dist,
            dir_offset_range, point_line_distance_range, loc_range, max_pending_distance, lambda_loc_range,
            loc_range_for_lambda_loc):
        """
        Decode locations and directions to adjacency matrix use Hungarian algorithm

        Notes:
            location: (batch, max_keypoint, 2)
            direction: (batch, max_keypoint, 6, 3)

        Args:
            distance_range: distance range for criterion adjustment
            point_line_distance: distance range between one keypoint and other matched keypoint's direction

        Returns:
            adjacency matrix of locations
        """
        assert location.shape[:2] == direction.shape[:2]
        locations = location.float()
        directions = copy.deepcopy(direction)
        batch_size, max_keypoint = location.shape[:2]

        adjacency_matrix = torch.zeros(batch_size, max_keypoint, max_keypoint)
        for batch in range(batch_size):
            adjacency_matrix[batch, :, :] = VectorizationDecoder.calculate_adjacency_matrix_hungarian(
                location=locations[batch], direction=directions[batch],
                lambda_angle_dist=lambda_angle_dist, lambda_point_line_dist=lambda_point_line_dist,
                dir_offset_range=dir_offset_range, point_line_distance_range=point_line_distance_range, 
                loc_range=loc_range, max_pending_distance=max_pending_distance, 
                lambda_loc_range=lambda_loc_range, loc_range_for_lambda_loc=loc_range_for_lambda_loc
            )

        return adjacency_matrix.int()

    @staticmethod
    def vectorization_decode_greedy(location, direction, lambda_angle_dist, lambda_point_line_dist,
                                    dir_offset_range, point_line_distance_range, loc_range, max_pending_distance,
                                    lambda_loc_range, loc_range_for_lambda_loc):
        """
        Decode locations and directions to adjacency matrix using greedy algorithm

        Notes:
            location: (batch, max_keypoint, 2)
            direction: (batch, max_keypoint, 6, 3)

        Args:
            distance_range: distance range for criterion adjustment
            point_line_distance: distance range between one keypoint and other matched keypoint's direction

        Returns:
            adjacency matrix of locations
        """
        assert location.shape[:2] == direction.shape[:2]
        locations = location.float()
        directions = copy.deepcopy(direction)
        batch_size, max_keypoint = location.shape[:2]

        adjacency_matrix = torch.zeros(batch_size, max_keypoint, max_keypoint)
        for batch in range(batch_size):
            adjacency_matrix[batch, :, :] = VectorizationDecoder.calculate_adjacency_matrix_greedy(
                location=locations[batch], direction=directions[batch],
                lambda_angle_dist=lambda_angle_dist, lambda_point_line_dist=lambda_point_line_dist,
                dir_offset_range=dir_offset_range, point_line_distance_range=point_line_distance_range,
                loc_range=loc_range, max_pending_distance=max_pending_distance,
                lambda_loc_range=lambda_loc_range, loc_range_for_lambda_loc=loc_range_for_lambda_loc
            )

        return adjacency_matrix.int()


class KeyPointDecoder:
    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        """
        Apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    @staticmethod
    def topk_score(scores, k=50):
        """ get top K point in score map """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_score, topk_index = torch.topk(scores.reshape(batch, channel, -1), k)  # (batch, channel, K)
        topk_index = topk_index % (height * width)  # (batch, channel, K)
        topk_ys = (topk_index / width).int().float()  # (batch, channel, K)
        topk_xs = (topk_index % width).int().float()  # (batch, channel, K)
        # # get all topk in in a batch
        # topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)  # (batch, K)
        #
        # # div by K because index is grouped by K(C x K shape)
        # topk_clses = (index / K).int()
        # topk_index = KeyPointDecoder.gather_feature(topk_index.view(batch, -1, 1), index).reshape(batch, K)
        # topk_ys = KeyPointDecoder.gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        # topk_xs = KeyPointDecoder.gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score.reshape(batch, k), topk_index.reshape(batch, k), \
               topk_ys.reshape(batch, k), topk_xs.reshape(batch, k)

    @staticmethod
    def gather_feature(feature_map, index, mask=None, use_transform=False):
        if use_transform:
            # change a (N, C, H, W) tenor to (N, HxW, C) shape
            batch, channel = feature_map.shape[:2]
            feature_map = feature_map.reshape(batch, channel, -1).permute((0, 2, 1))

        dim = feature_map.size(-1)
        index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
        feature_map = feature_map.gather(dim=1, index=index)
        if mask is not None:
            # this part is not called in Res18 dcn COCO
            mask = mask.unsqueeze(2).expand_as(feature_map)
            feature_map = feature_map[mask]
            feature_map = feature_map.reshape(-1, dim)
        return feature_map

    @staticmethod
    def find_endpoints(skeleton, coordinates):
        """
        Find endpoints in coordinates

        Notes:
            skeleton: (height, width)
            coordinates: [(y1, x1), (y2, x2)....]
        """
        skeleton = skeleton > 0
        height, width = skeleton.shape
        endpoints = []
        offset_pairs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for (y, x) in coordinates:
            neighbor_count = 0
            for (m, n) in offset_pairs:
                y_new, x_new = y + m, x + n
                if 0 <= y_new < height and 0 <= x_new < width:
                    if skeleton[y_new, x_new] == 1:
                        neighbor_count += 1
            if neighbor_count == 1:
                endpoints.append([y, x])

        return endpoints


class VectorizationDecoder:
    @staticmethod
    def calculate_adjacency_matrix_closest(
            location, direction, distance_range=[30, 448], point_line_distance=[50, 140]):
        """
        Calculate adjacency matrix according to location and direction

        Notes:
            location: (max_keypoint, 2)
            direction: (max_keypoint, 6, 3)

        Args:
            distance_range: distance range for criterion adjustment
            point_line_distance: distance range between one keypoint and other matched keypoint's direction
        """
        assert location.shape[0] == direction.shape[0]
        max_keypoint = location.shape[0]
        adjacency_matrix = torch.zeros(max_keypoint, max_keypoint)
        num_valid_keypoint = torch.all(location >= 0, dim=1).sum()
        valid_location = location[:num_valid_keypoint]
        distance_matrix = torch.norm(valid_location.unsqueeze(1) - valid_location.unsqueeze(0), dim=-1)
        pairs_index = VectorizationDecoder.argsort_matrix(distance_matrix)
        for (i, j) in pairs_index:
            if i < j:
                # only cope with upper right of the matrix
                loc1 = location[i]  # (2)
                loc2 = location[j]  # (2)
                dir1 = direction[i]  # (6, 3)
                dir2 = direction[j]  # (6, 3)

                best_match_idx = VectorizationDecoder.compare_direction(
                    dir1=dir1, loc1=loc1, dir2=dir2, loc2=loc2,
                    min_distance=distance_range[0], max_distance=distance_range[1],
                    min_point_line_distance=point_line_distance[0],
                    max_point_line_distance=point_line_distance[1]
                )

                if best_match_idx is not None:
                    # remove matched direction
                    dir1_idx, dir2_idx = best_match_idx
                    direction[i, dir1_idx] = 0
                    direction[j, dir2_idx] = 0
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

        return adjacency_matrix

    @staticmethod
    def calculate_adjacency_matrix_hungarian(
            location, direction, lambda_angle_dist, lambda_point_line_dist,
            dir_offset_range, point_line_distance_range, loc_range, max_pending_distance,
            lambda_loc_range, loc_range_for_lambda_loc):
        """
        Calculate adjacency matrix according to location and direction, using Hungarian Algorithm

        Args:
            location (torch.Tensor): (max_keypoint, 2)
            direction (torch.Tensor): (max_keypoint, 6, 3)

        Returns:
            adjacency_matrix of these keypoints
        """

        assert location.shape[0] == direction.shape[0]
        max_keypoint = location.shape[0]
        adjacency_matrix = torch.zeros(max_keypoint, max_keypoint)
        num_valid_keypoint = torch.all(location >= 0, dim=1).sum()

        # First, we have to save all the direction info in a list and each dir as a tuple
        # (location, direction, sec_dix, loc_idx)
        single_dir_info_list = []
        for loc_idx in range(num_valid_keypoint):
            for dir_idx in range(6):
                if direction[loc_idx, dir_idx, 0] > 0:
                    dir_angle = VectorizationDecoder.get_degree(
                        direction[loc_idx, dir_idx, 1], direction[loc_idx, dir_idx, 2]
                    )
                    sec_idx = int(dir_angle) // 60
                    single_dir_info = (location[loc_idx], direction[loc_idx, dir_idx, 1:], sec_idx, loc_idx)
                    single_dir_info_list.append(single_dir_info)

        # Next, we try to pick out all the possible edges (two direction could form an edge)
        dir_count = len(single_dir_info_list)
        cost_matrix = np.full((dir_count, dir_count), 1000.)
        for i in range(dir_count):
            current_dir_info = single_dir_info_list[i]
            possible_matching_sec_list = \
                [(current_dir_info[2] + 2) % 6, (current_dir_info[2] + 3) % 6, (current_dir_info[2] + 4) % 6]
            for j in range(i + 1, dir_count):
                # if True:
                if single_dir_info_list[j][2] in possible_matching_sec_list:
                    # we only take adjacent section idx as possible sections
                    to_match_dir_info = single_dir_info_list[j]
                    if torch.all(to_match_dir_info[0] == current_dir_info[0]):
                        # we don't let two directions of same keypoint to be matched
                        continue
                    # we calculate distance of two directions to judge whether to match
                    distance = VectorizationDecoder.calculate_distance_of_two_directions(
                        dir_info1=current_dir_info, dir_info2=to_match_dir_info,
                        lambda_angle_dist=lambda_angle_dist, lambda_point_line_dist=lambda_point_line_dist,
                        dir_offset_range=dir_offset_range, point_line_distance_range=point_line_distance_range, 
                        loc_range=loc_range, lambda_loc_range=lambda_loc_range, loc_range_for_lambda_loc=loc_range_for_lambda_loc
                    )
                    if distance <= max_pending_distance:
                        cost_matrix[i, j] = distance
                        cost_matrix[j, i] = distance

        # Finally, use Hungarian Algorithm to compute the best matching strategy
        row, col = linear_sum_assignment(cost_matrix)  # Hungarian matching
        match_output = [(dir_idx_1, dir_idx_2) for dir_idx_1, dir_idx_2 in zip(row, col)]
        # TODO: may use scipy.sparse.csgraph.min_weight_full_bipartite_matching
        for dir_idx_1, dir_idx_2 in match_output:
            if cost_matrix[dir_idx_1, dir_idx_2] < 500:
                loc_idx1 = single_dir_info_list[dir_idx_1][3]
                loc_idx2 = single_dir_info_list[dir_idx_2][3]
                assert loc_idx1 != loc_idx2
                adjacency_matrix[loc_idx1, loc_idx2] = 1
                adjacency_matrix[loc_idx2, loc_idx1] = 1

        return adjacency_matrix

    @staticmethod
    def calculate_adjacency_matrix_greedy(
            location, direction, lambda_angle_dist, lambda_point_line_dist,
            dir_offset_range, point_line_distance_range, loc_range, max_pending_distance,
            lambda_loc_range, loc_range_for_lambda_loc
    ):
        """
        Calculate adjacency matrix according to location and direction, using Greedy Algorithm

        Args:
            location (torch.Tensor): (max_keypoint, 2)
            direction (torch.Tensor): (max_keypoint, 6, 3)

        Returns:
            adjacency_matrix of these keypoints
        """
        assert location.shape[0] == direction.shape[0]
        max_keypoint = location.shape[0]
        adjacency_matrix = torch.zeros(max_keypoint, max_keypoint)
        num_valid_keypoint = torch.all(location >= 0, dim=1).sum()

        # First, we have to save all the direction info in a list and each dir as a tuple
        # (location, direction, sec_dix, loc_idx)
        single_dir_info_list = []
        for loc_idx in range(num_valid_keypoint):
            for dir_idx in range(6):
                if direction[loc_idx, dir_idx, 0] > 0:
                    dir_angle = VectorizationDecoder.get_degree(
                        direction[loc_idx, dir_idx, 1], direction[loc_idx, dir_idx, 2]
                    )
                    sec_idx = int(dir_angle) // 60
                    single_dir_info = (location[loc_idx], direction[loc_idx, dir_idx, 1:], sec_idx, loc_idx)
                    single_dir_info_list.append(single_dir_info)

        # Next, we try to pick out all the possible edges (two direction could form an edge)
        dir_count = len(single_dir_info_list)
        cost_matrix = np.full((dir_count, dir_count), 1000.)
        for i in range(dir_count):
            current_dir_info = single_dir_info_list[i]
            possible_matching_sec_list = \
                [(current_dir_info[2] + 2) % 6, (current_dir_info[2] + 3) % 6, (current_dir_info[2] + 4) % 6]
            for j in range(i + 1, dir_count):
                # if True:
                if single_dir_info_list[j][2] in possible_matching_sec_list:
                    # we only take adjacent section idx as possible sections
                    to_match_dir_info = single_dir_info_list[j]
                    if torch.all(to_match_dir_info[0] == current_dir_info[0]):
                        # we don't let two directions of same keypoint to be matched
                        continue
                    # we calculate distance of two directions to judge whether to match
                    distance = VectorizationDecoder.calculate_distance_of_two_directions(
                        dir_info1=current_dir_info, dir_info2=to_match_dir_info,
                        lambda_angle_dist=lambda_angle_dist, lambda_point_line_dist=lambda_point_line_dist,
                        dir_offset_range=dir_offset_range, point_line_distance_range=point_line_distance_range,
                        loc_range=loc_range, lambda_loc_range=lambda_loc_range,
                        loc_range_for_lambda_loc=loc_range_for_lambda_loc
                    )
                    if distance <= max_pending_distance:
                        cost_matrix[i, j] = distance
                        cost_matrix[j, i] = distance

        # Finally, use Greedy Algorithm to compute the best matching strategy
        match_output = VectorizationDecoder.greedy(cost_matrix)  # Hungarian matching
        for dir_idx_1, dir_idx_2 in match_output:
            loc_idx1 = single_dir_info_list[dir_idx_1][3]
            loc_idx2 = single_dir_info_list[dir_idx_2][3]
            assert loc_idx1 != loc_idx2
            adjacency_matrix[loc_idx1, loc_idx2] = 1
            adjacency_matrix[loc_idx2, loc_idx1] = 1

        return adjacency_matrix

    
    @staticmethod
    def greedy(cost_matrix):
        """
        Find pairs with lowest cost using greedy algorithm
        Args:
            cost_matrix (np.ndarray): input cost matrix, 1000 denotes invalid value

        Returns:
            match_pairs (List(Tuple(int, int))): best matching pairs
        """
        match_pairs = []
        if len(cost_matrix) > 0:
            lowest_value = np.min(cost_matrix)
            find_smallest = lambda arr: np.unravel_index(np.argmin(arr), arr.shape)
            while lowest_value < 1000:
                idx1, idx2 = find_smallest(cost_matrix)
                match_pairs.append((idx1, idx2))
                cost_matrix[idx1, :] = 1000
                cost_matrix[idx2, :] = 1000
                cost_matrix[:, idx1] = 1000
                cost_matrix[:, idx2] = 1000
                lowest_value = np.min(cost_matrix)

        return match_pairs


    @staticmethod
    def argsort_matrix(matrix, descending=False):
        """
        Give a matrix, get the sorted coordinate (min->max or max->min)

        Notes:
            matrix: (height, width)

        Args:
            descending: False means min --> max, True means max --> min
        """
        height, width = matrix.shape
        flat_sort = torch.argsort(matrix.flatten(), descending=descending)
        coord_width = flat_sort % width
        coord_height = torch.div(flat_sort, width, rounding_mode='trunc')
        return torch.stack([coord_height, coord_width], dim=1)

    @staticmethod
    def compare_direction(dir1, loc1, dir2, loc2, min_distance=30., max_distance=448.,
                          min_point_line_distance=50., max_point_line_distance=140.):
        """
        compare if keypoint1 can match keypoint2

        Notes:
            dir1: (6, 3)
            loc1: (2)
            dir2: (6, 3)
            loc2: (2)

        Returns:
            Matched direction index in compared_dir, else None;
            Distance of two matched dir, else None
        """
        # calculate threshold
        loc_distance = torch.norm(loc1 - loc2)
        if loc_distance <= min_distance:
            threshold_point_line_distance = min_point_line_distance
        elif loc_distance >= max_distance:
            threshold_point_line_distance = max_point_line_distance
        else:
            threshold_point_line_distance = \
                min_point_line_distance + ((loc_distance - min_distance) / (max_distance - min_distance)) * (max_point_line_distance - min_point_line_distance)

        distance_matrix = torch.full((6, 6), fill_value=1e6)
        for i in range(6):
            if dir1[i, 0] > 0:
                for j in range(6):
                    if dir2[j, 0] > 0:
                        tem_dir1 = torch.tensor([dir1[i, 2], dir1[i, 1]])
                        point_line_distance1 = VectorizationDecoder.get_point_line_distance(loc2, loc1, tem_dir1)
                        tem_dir2 = torch.tensor([dir2[j, 2], dir2[j, 1]])
                        point_line_distance2 = VectorizationDecoder.get_point_line_distance(loc1, loc2, tem_dir2)
                        if point_line_distance1 + point_line_distance2 <= threshold_point_line_distance:
                            distance_matrix[i, j] = torch.tensor(point_line_distance1) + torch.tensor(point_line_distance2)

        best_match_pair = VectorizationDecoder.argsort_matrix(distance_matrix)[0]
        if distance_matrix[best_match_pair[0], best_match_pair[1]] >= 1e6:
            return None
        else:
            return best_match_pair

    @staticmethod
    def get_unit_vector(start_loc, end_loc):
        """
        Compute unit vector.

        Notes:
            start_loc: (2)
            end_loc: (2)
        """
        return (end_loc.float() - start_loc) / torch.clamp(torch.norm(end_loc.float() - start_loc), min=1e-8)

    @staticmethod
    def get_angle_between_two_vectors(vec1, vec2):
        """
        Compute angle between two vector, [0, 180].

        Notes:
            vec1: (2)
            vec2: (2)
        """
        cos_theta = torch.sum(vec1 * vec2) / torch.clamp(torch.norm(vec1) * torch.norm(vec2), min=1e-8)
        # must clamp cos_theta to avoid float error!!!
        cos_theta = torch.clamp(cos_theta, min=-1., max=1.)
        degree = torch.rad2deg(torch.arccos(cos_theta))
        return degree

    @staticmethod
    def get_point_line_distance(point_loc, dir_loc, dir):
        """ Compute vertical distance between point and dir at dir_loc"""
        c = point_loc - dir_loc
        a = dir
        # len_b = torch.sqrt(torch.clamp(torch.norm(c) ** 2 - (torch.sum(a * c) / torch.norm(a)) ** 2, min=0.))
        len_b = np.linalg.norm(np.cross(c, a) / np.linalg.norm(a))
        if np.dot(c, a) < 0:
            len_b = 10000
        return len_b


    @staticmethod
    def find_best_match_loc_and_idx(current_loc, candidate_keypoint):
        """
        Find best match keypoint location and direction index

        Notes:
            current_loc: (2)
            candidate_keypoint: [(loc, dir_distance, loc_idx, dir_idx), ...]

        Returns:
            best_match_loc_idx, best_match_dir_idx
        """
        loc_distance = [torch.norm(current_loc - keypoint[0]) for keypoint in candidate_keypoint]
        min_loc_distance_index_list = [i for i, x in enumerate(loc_distance) if x == min(loc_distance)]
        if len(min_loc_distance_index_list) > 1:
            candidate_keypoint = [candidate_keypoint[idx] for idx in min_loc_distance_index_list]
            dir_distance = [keypoint[1] for keypoint in candidate_keypoint]
            min_dir_distance_index = dir_distance.index(min(dir_distance))
            best_match_loc_idx = candidate_keypoint[min_dir_distance_index][2]
            best_match_dir_idx = candidate_keypoint[min_dir_distance_index][3]
        else:
            best_match_loc_idx = candidate_keypoint[min_loc_distance_index_list[0]][2]
            best_match_dir_idx = candidate_keypoint[min_loc_distance_index_list[0]][3]
        return best_match_loc_idx, best_match_dir_idx

    @staticmethod
    def calculate_distance(loc_x, dir_x, loc_y, dir_y, input_shape):
        """ Calculate shortest designed distance between x and y

        Note:
            loc_x, loc_y: (2)
            dir_x, dir_y: (6, 3)

        Return:
            shortest distance between x and y
        """
        max_loc = min(input_shape)

        loc_distance = torch.norm((loc_x - loc_y) / max_loc)
        dir_distance = []
        for i in range(6):
            for j in range(6):
                if dir_x[i, 0] > 0 and dir_y[j, 0] > 0:
                    dir_distance.append(
                        VectorizationDecoder.designed_angle_distance(dir_x[i, 1:], dir_y[j, 1:])
                    )

        return 100 * loc_distance * min(dir_distance)

    @staticmethod
    def designed_angle_distance(x, y):
        """ Calculate designed distance between x and y """
        return torch.norm(x + y)

    @staticmethod
    def get_degree(sin_value, cos_value):
        """ get angle in [0, 360) via sin and cos value """
        theta = math.atan2(sin_value, cos_value)
        if theta < 0:
            theta += 2 * math.pi
        return math.degrees(theta)

    @staticmethod
    def calculate_distance_of_two_directions(
            dir_info1, dir_info2, lambda_angle_dist, lambda_point_line_dist,
            dir_offset_range, point_line_distance_range, loc_range, lambda_loc_range,loc_range_for_lambda_loc
    ):
        """
        Compute distance between two directions, used to judge whether can form an edge

        Args:
            dir_info1 (Tuple): (location, direction, sec_dix, loc_idx)
            dir_info2 (Tuple): (location, direction, sec_dix, loc_idx)
            lambda_angle_dist (float): weight of angle distance
            lambda_point_line_dist (float): weight of point line distance
            max_dir_offset (float): threshold for angle
            max_point_line_distance (float): threshold for point line distance
            loc_range (List): 
            lambda_loc_range (float): 

        Returns:
            Designed distance of two directions
        """
        loc1, dir1, _, _ = dir_info1
        loc2, dir2, _, _ = dir_info2

        # unify coordinate system of reference_vec and direction !
        reference_vec1 = VectorizationDecoder.get_unit_vector(start_loc=loc1, end_loc=loc2)
        tem_dir1 = torch.as_tensor([dir1[1], dir1[0]])
        angle1 = VectorizationDecoder.get_angle_between_two_vectors(reference_vec1, tem_dir1)
        point_line_distance1 = VectorizationDecoder.get_point_line_distance(loc2, loc1, tem_dir1)

        reference_vec2 = VectorizationDecoder.get_unit_vector(start_loc=loc2, end_loc=loc1)
        tem_dir2 = torch.as_tensor([dir2[1], dir2[0]])
        angle2 = VectorizationDecoder.get_angle_between_two_vectors(reference_vec2, tem_dir2)
        point_line_distance2 = VectorizationDecoder.get_point_line_distance(loc1, loc2, tem_dir2)

        loc_distance = torch.norm(loc1 - loc2)
        
        if loc_distance <= loc_range[0]:
            max_point_line_distance = point_line_distance_range[0]
            max_dir_offset = dir_offset_range[0]
        elif loc_distance >= loc_range[1]:
            max_point_line_distance = point_line_distance_range[1]
            max_dir_offset = dir_offset_range[1]
        else:
            max_point_line_distance = \
                point_line_distance_range[0] + ((loc_distance - loc_range[0]) / (loc_range[1] - loc_range[0])) * \
                (point_line_distance_range[1] - point_line_distance_range[0])
            max_dir_offset = dir_offset_range[0] + ((loc_distance - loc_range[0]) / (loc_range[1] - loc_range[0])) * \
                (dir_offset_range[1] - dir_offset_range[0])

        if angle1 + angle2 >= max_dir_offset or \
            point_line_distance1 + point_line_distance2 >= max_point_line_distance:
            return 1000

        # we have to compute another coefficient by location distance, where closer is better
        if loc_distance < loc_range_for_lambda_loc[0]:
            lambda_loc = lambda_loc_range[0]
        elif loc_distance > loc_range_for_lambda_loc[1]:
            lambda_loc = lambda_loc_range[1]
        else:
            lambda_loc = lambda_loc_range[0] + ((loc_distance - loc_range_for_lambda_loc[0]) / (loc_range_for_lambda_loc[1] - loc_range_for_lambda_loc[0])) * \
                (lambda_loc_range[1] - lambda_loc_range[0])
            
        distance = (point_line_distance1 + point_line_distance2) * lambda_loc

        return distance

