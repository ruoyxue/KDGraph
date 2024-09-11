import copy
import cv2
import networkx as nx
import numpy as np
import sympy
import torch
import torch.nn.functional as F
from crdp import rdp
from scipy.optimize import linear_sum_assignment
from skimage import draw
from sympy import Point
from .decoder import Decoder, KeyPointDecoder
from scipy.spatial import cKDTree


@torch.no_grad()
def patch_inference(data, model, config, need_keypoint=False):
    """
    Inference small patches and stitch them to a large image

    Notes:
        data: (batch, 3, height, width)

    Returns:
        pred_centerlines: (batch, height, width) np.array
        pred_graphs: [graph1, graph2, ...]  graph is networkx.Graph
        whole_locations: (batch, max_keypoint, 2)
        whole_directions: (batch, max_keypoint, 6, 3)
    """
    crop_size = config["VAL"]["CROP_SIZE"]
    overlap_size = config["VAL"]["OVERLAP_SIZE"]
    assert overlap_size % 2 == 0
    half_overlap_size = overlap_size // 2

    batch_size, _, height, width = data.shape

    if config['TRAIN']['LOSS']['CRITERION_LOC'] in ["weighted_bceloss", "weighted_modifiedfocalloss"] or \
            config['TRAIN']['LOSS']['CRITERION_DIR_PROB'] in ["weighted_bceloss", "weighted_modifiedfocalloss"]:
        use_sigmoid = True
    else:
        use_sigmoid = False

    # patch image and stitch
    whole_graph_list = [nx.Graph() for _ in range(batch_size)]  # record whole image graphs
    whole_patch_intersection_list = []  # record intersection line of two patches
    whole_patch_mask = torch.zeros((height, width), dtype=torch.int)  # record different patch areas

    ######################################################################################
    # Step 1: infer image patches
    ######################################################################################
    # generate left-upper coordinates of image patches
    x_list = list(range(0, width - crop_size, crop_size - overlap_size)) + [width - crop_size]
    y_list = list(range(0, height - crop_size, crop_size - overlap_size)) + [height - crop_size]
    left_upper_coord_list = [(x, y) for y in y_list for x in x_list]
    patch_max_keypoint = config["OTHER_ARGS"]["MAX_KEYPOINT"]
    whole_locations = torch.full((batch_size, patch_max_keypoint * len(left_upper_coord_list), 2), -1.)
    whole_directions = torch.full((batch_size, patch_max_keypoint * len(left_upper_coord_list), 6, 3), 0.)

    patch_count = 0
    for x, y in left_upper_coord_list:
        valid_mask = (whole_patch_mask[y: y + crop_size, x: x + crop_size]) == 0  # used to avoid overlapping

        if x != 0:
            valid_mask[:, :half_overlap_size] = 0
        if x != width - crop_size:
            valid_mask[:, crop_size - half_overlap_size:] = 0
        if y != 0:
            valid_mask[:half_overlap_size, :] = 0
        if y != height - crop_size:
            valid_mask[crop_size - half_overlap_size:, :] = 0

        whole_patch_mask[y: y + crop_size, x: x + crop_size] += valid_mask
        data_patch = data[:, :, y: y + crop_size, x: x + crop_size]
        loc_feat_patch, dir_feat_patch = model(data_patch)

        loc_feat_patch = loc_feat_patch.cpu()
        dir_feat_patch = dir_feat_patch.cpu()
        ############################################################
        # Step 2: link keypoints in small patch
        ############################################################
        # Step 2.1: small patch keypoint decode
        
        if use_sigmoid:
            # reduce keypoint features around the edge
            loc_feat_patch = torch.sigmoid(loc_feat_patch)
            edge_len = 10 + config["OTHER_ARGS"]["RADIUS"]
            edge_reduce_feature = 0.1
            batch_size, _, patch_height, patch_width = loc_feat_patch.shape
            edge_mask = torch.zeros((batch_size, 1, patch_height, patch_width))
            edge_mask[:, :, :edge_len, :] = 1
            edge_mask[:, :, patch_height - edge_len: patch_height, :] = 1
            edge_mask[:, :, :, :edge_len] = 1
            edge_mask[:, :, :, patch_width - edge_len: patch_width] = 1
            loc_feat_patch[edge_mask == 1] -= edge_reduce_feature
            
            dir_feat_patch[:, [0, 3, 6, 9, 12, 15], :, :] = torch.sigmoid(dir_feat_patch[:, [0, 3, 6, 9, 12, 15], :, :])
            edge_dir_prob_reduce = 0.1
            edge_dir_prob_mask = torch.zeros((batch_size, 18, patch_height, patch_width))
            edge_dir_prob_mask[:, 0, :, :] = edge_mask.squeeze(1)
            edge_dir_prob_mask[:, 3, :, :] = edge_mask.squeeze(1)
            edge_dir_prob_mask[:, 6, :, :] = edge_mask.squeeze(1)
            edge_dir_prob_mask[:, 9, :, :] = edge_mask.squeeze(1)
            edge_dir_prob_mask[:, 12, :, :] = edge_mask.squeeze(1)
            edge_dir_prob_mask[:, 15, :, :] = edge_mask.squeeze(1)
            dir_feat_patch[edge_dir_prob_mask == 1] -= edge_dir_prob_reduce

        loc_feat_patch = KeyPointDecoder.pseudo_nms(loc_feat_patch, pool_size=config["OTHER_ARGS"]["NMS_POOLSIZE"])
        scores, ind, patch_ys, patch_xs = KeyPointDecoder.topk_score(loc_feat_patch, config["OTHER_ARGS"]["MAX_KEYPOINT"])
        # remove low confidence keypoints
        patch_locations = torch.zeros(batch_size, config["OTHER_ARGS"]["MAX_KEYPOINT"], 2)
        copy_patch_locations = torch.zeros(batch_size, config["OTHER_ARGS"]["MAX_KEYPOINT"], 2)
        row, col = torch.where(scores < config["OTHER_ARGS"]["LOC_THRESHOLD"])
        patch_ys[[row, col]], patch_xs[[row, col]] = -1, -1
        patch_locations[:, :, 0] = patch_xs
        patch_locations[:, :, 1] = patch_ys
        copy_patch_locations[:, :, 0] = patch_xs + x
        copy_patch_locations[:, :, 1] = patch_ys + y
        
        patch_directions = KeyPointDecoder.gather_feature(dir_feat_patch, ind, use_transform=True)
        patch_directions = patch_directions.reshape(batch_size, config["OTHER_ARGS"]["MAX_KEYPOINT"], 6, 3)
        # normalise direction vector
        length = torch.clamp(torch.sqrt(patch_directions[:, :, :, 1] ** 2 + patch_directions[:, :, :, 2] ** 2), 1e-8)
        patch_directions[:, :, :, 1] /= length
        patch_directions[:, :, :, 2] /= length
        patch_directions[patch_directions[:, :, :, 0] < config["OTHER_ARGS"]["DIR_THRESHOLD"]] = 0
        
        for _ in range(3):
            patch_directions = clean_direction(patch_directions, clean_dir_offset=30)
        
        whole_locations[:, patch_count * patch_max_keypoint:(patch_count + 1) * patch_max_keypoint, :] = copy_patch_locations
        whole_directions[:, patch_count * patch_max_keypoint:(patch_count + 1) * patch_max_keypoint, :, :] = patch_directions
        
        # Step 2.2: small patch link prediction
        adjacency_matrix = Decoder.vectorization_decode_greedy(
            location=patch_locations, direction=patch_directions,
            lambda_angle_dist=0, lambda_point_line_dist=1,
            dir_offset_range=[44.99, 45], point_line_distance_range=[70, 140],
            loc_range=[0, int(crop_size)], max_pending_distance=1000, lambda_loc_range=[0, 1],
            loc_range_for_lambda_loc=[0, int(crop_size)]
        )

        # add edges inside small patch to patch_graph and update whole_graph_list
        for batch in range(batch_size):
            patch_graph = nx.Graph()
            row, col = torch.where(adjacency_matrix[batch] > 0)
            valid_link_list = [(r, c) for r, c, in zip(row, col)]
            for i, j in valid_link_list:
                if i < j:
                    # cannot use tensor here !!!
                    point1_x = int(patch_locations[batch][i][0])
                    point1_y = int(patch_locations[batch][i][1])
                    point2_x = int(patch_locations[batch][j][0])
                    point2_y = int(patch_locations[batch][j][1])
                    patch_graph.add_edge(
                        (point1_x, point1_y), (point2_x, point2_y)
                    )

            # cut link that outside valid_mask and add new node at the edge
            patch_graph = cut_graph_edge_outside_mask(patch_graph, valid_mask)

            # transform small patch locs to world locs
            for A, B in patch_graph.edges():
                w_A = (A[0] + x, A[1] + y)
                w_B = (B[0] + x, B[1] + y)
                whole_graph_list[batch].add_edge(w_A, w_B)

        # Step 2.3: save information of patches
        # save down and right intersection line into whole_patch_intersection_list
        rec_vertices, _ = cv2.findContours(np.uint8(valid_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_upper, left_lower, right_lower, right_upper = \
            rec_vertices[0][0][0], rec_vertices[0][1][0], \
            rec_vertices[0][2][0], rec_vertices[0][3][0]  # save as (x, y)
        left_lower = np.array([left_lower[0] + x, left_lower[1] + y])
        right_lower = np.array([right_lower[0] + x, right_lower[1] + y])
        right_upper = np.array([right_upper[0] + x, right_upper[1] + y])
        if not (right_lower[1] == height - 1):
            whole_patch_intersection_list.append((left_lower, right_lower))  # save down line
        if not (right_lower[0] == width - 1):
            whole_patch_intersection_list.append((right_upper, right_lower))  # save right line
        patch_count += 1

    ######################################################################################
    # Step 3: simplify graph
    ######################################################################################
    # simplify keypoints that around patches intersection lines
    assert torch.sum(whole_patch_mask == 0) == 0
    for batch in range(batch_size):
        # simplify keypoints near the intersection of patches
        for _ in range(3):
            whole_graph_list[batch] = simplify_graph_around_patch_intersection(
                whole_graph_list[batch], whole_patch_intersection_list,
                buffer_size=5, max_match_dist=35, shape=(height, width)
            )
        
        # refine graph by connecting adjacent endpoints
        for _ in range(3):
            whole_graph_list[batch] = refine_graph(
                whole_graph_list[batch], (height, width), max_connect_dist=50, edge_ignore_size=10
            )

        # simplify graph using rdp
        whole_graph_list[batch] = simplify_graph_rdp(whole_graph_list[batch], rdp_dist=3)

        # remove small segments
        whole_graph_list[batch] = remove_small_segments(whole_graph_list[batch], remove_length=150)

        # merge dense keypoints
        for _ in range(3):
            whole_graph_list[batch] = merge_dense_keypoints(whole_graph_list[batch], merge_dist=15)
    
    whole_centerline_list = get_centerline(whole_graph_list, (height, width))
    return whole_centerline_list, whole_graph_list, whole_locations, whole_directions


def pseudo_nms(fmap, pool_size=3):
    """ Apply max pooling to get the same effect of nms """
    pad = (pool_size - 1) // 2
    fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
    keep = (fmap_max == fmap).float()
    return fmap * keep


def remove_small_segments(graph, remove_length=50):
    """
    Remove subgraphs whose overall length <= remove_length

    Args:
        graph (networkx.Graph): input graph
        remove_length (float): threshold for removing subgraph
    """
    nodes_to_remove = []
    for sub_graph_nodes in nx.connected_components(graph):
        sub_graph = graph.subgraph(sub_graph_nodes)
        length = 0
        for node_1, node_2 in sub_graph.edges():
            length += np.linalg.norm(np.array(node_1) - np.array(node_2))
        if length <= remove_length:
            nodes_to_remove.extend(list(sub_graph.nodes()))
    
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def node_dist(node1, node2):
    """
    Compute Euclidean distance of two nodes

    Args:
        node1 (tuple): (x, y) namely (col, row)
        node2 (tuple): (x, y) namely (col, row)

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(np.array(node1) - np.array(node2))


def merge_dense_keypoints(graph, merge_dist=5):
    """
    Merge keypoints that are close to each other to a point

    Args:
        graph (networkx.Graph): input graph
        merge_dist (float): max dist to merge

    Returns:
        merged graph
    """
    done_set = []  # record nodes that have been searched
    match_dict = {}  # record points to be merged and their centroid

    # compute merge info
    for current_node in graph.nodes:
        if current_node not in done_set:
            neighbors = graph.neighbors(current_node)
            node_to_merge = [current_node]
            for neighbor_node in neighbors:
                if neighbor_node not in done_set and node_dist(current_node, neighbor_node) <= merge_dist:
                    node_to_merge.append(neighbor_node)

            if len(node_to_merge) >= 2:
                # compute centroid and save info in dict
                centroid = tuple(np.round(np.mean(node_to_merge, axis=0)).astype(int))
                match_dict[centroid] = node_to_merge

            done_set.extend(node_to_merge)

    # start to merge
    for center_keypoint, graph_keypoints_list in match_dict.items():
        node_to_remove = []
        edge_to_add = []
        for keypoint in graph_keypoints_list:
            for neighbor in copy.deepcopy(graph.neighbors(keypoint)):
                edge_to_add.append((neighbor, center_keypoint))
            node_to_remove.append(keypoint)

        while True:
            try:
                node_to_remove.remove(center_keypoint)
            except:
                break
        graph.add_edges_from(edge_to_add)
        graph.remove_nodes_from(node_to_remove)

    # remove isolated nodes
    nodes = copy.deepcopy(graph.nodes())
    for node in nodes:
        if len(list(graph.neighbors(node))) == 0:
            graph.remove_node(node)

    return graph


def clean_edge_direction(locations, directions, patch_shape, edge_width=20, clean_dir_offset=15.):
    """
    Remove directions that are close to each other at the edge, and only retain the direction with max probability

    Args:
        locations (torch.Tensor): (batch_size, max_keypoint, 2)
        directions (torch.Tensor):  (batch_size, max_keypoint, 6, 3)
        clean_dir_offset (float): threshold for clean
    """
    batch_size, max_keypoint, _, _ = directions.shape
    height, width = patch_shape
    cos_threshold = torch.cos(torch.deg2rad(torch.as_tensor([clean_dir_offset])))

    for batch_idx in range(batch_size):
        for keypoint_idx in range(max_keypoint):
            loc_x, loc_y = locations[batch_idx, keypoint_idx]
            if not ((edge_width <= loc_x <= width - edge_width) and (edge_width <= loc_y <= height - edge_width)) \
                and loc_x >=0 and loc_y >= 0:
                valid_direction_index_list = list(np.array(torch.where(directions[batch_idx, keypoint_idx, :, 0] > 0)[0]))
                # find direction index that to be cleaned
                pairs_list = []
                done_list = []
                for i in valid_direction_index_list:
                    current_dir = directions[batch_idx, keypoint_idx, i]
                    match_dir_list = [i]
                    for j in valid_direction_index_list:
                        if j not in done_list and i != j:
                            other_dir = directions[batch_idx, keypoint_idx, j]
                            cos_theta = torch.sum(current_dir * other_dir) / (torch.norm(current_dir) * torch.norm(other_dir))
                            if cos_theta >= cos_threshold:
                                match_dir_list.append(j)

                    if len(match_dir_list) > 0:
                        done_list.extend(match_dir_list)
                        pairs_list.append(match_dir_list)
                
                # clean paired directions -- only retain dir with highest probability
                for pair in pairs_list:
                    max_prob_index = pair[torch.argmax(directions[batch_idx, keypoint_idx, pair, 0])]
                    for i in pair:
                        if i != max_prob_index:
                            directions[batch_idx, keypoint_idx, i, 0] = 0.            
            
    return directions


def clean_direction(directions, clean_dir_offset=15.):
    """
    Remove directions that are close to each other, and only retain the direction with max probability

    Args:
        directions (torch.Tensor):  (batch_size, max_keypoint, 6, 3)
        clean_dir_offset (float): threshold for clean
    """
    batch_size, max_keypoint, _, _ = directions.shape
    cos_threshold = torch.cos(torch.deg2rad(torch.as_tensor([clean_dir_offset])))

    for batch_idx in range(batch_size):
        for keypoint_idx in range(max_keypoint):
            valid_direction_index_list = list(np.array(torch.where(directions[batch_idx, keypoint_idx, :, 0] > 0)[0]))
            # random.shuffle(valid_direction_index_list)
            
            # find direction index that to be cleaned
            pairs_list = []
            done_list = []
            for i in valid_direction_index_list:
                current_dir = directions[batch_idx, keypoint_idx, i]
                match_dir_list = [i]
                for j in valid_direction_index_list:
                    if j not in done_list and i != j:
                        other_dir = directions[batch_idx, keypoint_idx, j]
                        cos_theta = torch.sum(current_dir * other_dir) / (torch.norm(current_dir) * torch.norm(other_dir))
                        if cos_theta >= cos_threshold:
                            match_dir_list.append(j)

                if len(match_dir_list) > 0:
                    done_list.extend(match_dir_list)
                    pairs_list.append(match_dir_list)
            
            # clean paired directions -- only retain dir with highest probability
            for pair in pairs_list:
                max_prob_index = pair[torch.argmax(directions[batch_idx, keypoint_idx, pair, 0])]
                for i in pair:
                    if i != max_prob_index:
                        directions[batch_idx, keypoint_idx, i, 0] = 0.            
            
    return directions


def simplify_graph_around_patch_intersection(graph, intersection_list, buffer_size, max_match_dist, shape):
    """
    Simplify keypoints near the intersection of patches

    Args:
        graph: networkx.Graph
        intersection_list: denote where is the intersection
        buffer_size: the width of the buffer
        max_match_dist: max distance for two nodes to be matched
        shape: image shape
    """
    whole_match_dict = {}
    intersection_mask = np.zeros(shape, dtype=np.uint8)
    edge_point_list = []

    for a, b in intersection_list:
        cv2.line(intersection_mask, a, b, color=(255, 255, 255), thickness=buffer_size * 2)

    for node in graph.nodes():
        if intersection_mask[node[1], node[0]] > 0:
            edge_point_list.append(node)

    if len(edge_point_list) > 1:
        point_tree = cKDTree(edge_point_list)
        candidate_pairs = point_tree.query_ball_tree(point_tree, r=max_match_dist)
        done_set = set()

        current_point_idx = 0
        for connect_list in candidate_pairs:
            point_to_connect = list(set(connect_list).difference(done_set))
            if current_point_idx in point_to_connect:
                if len(point_to_connect) >= 2:
                    # have to connect points in point_to_connect
                    points = [edge_point_list[i] for i in point_to_connect]
                    centroid = tuple(np.round(np.mean(points, axis=0)).astype(int))
                    whole_match_dict[centroid] = points
                done_set.update(point_to_connect)
            current_point_idx += 1

    for new_keypoint in whole_match_dict.keys():
        node_to_remove = []
        edge_to_add = []
        # only update those keypoints that have edges in both patches
        graph_keypoints_list = whole_match_dict[new_keypoint]
        for keypoint in graph_keypoints_list:
            for neighbor in copy.deepcopy(graph.neighbors(keypoint)):
                edge_to_add.append((neighbor, new_keypoint))
            node_to_remove.append(keypoint)
        if new_keypoint in node_to_remove:
            node_to_remove.remove(new_keypoint)
        graph.add_edges_from(edge_to_add)
        graph.remove_nodes_from(node_to_remove)

    return graph


def get_centerline(graph_list, image_shape):
    """ draw centerline according to graph_list"""
    centerline_list = []
    for i in range(len(graph_list)):
        graph = graph_list[i]
        centerline = np.zeros((image_shape[0], image_shape[1]))
        for node_A, node_B in graph.edges():
            node_A = (np.int(node_A[0]), np.int(node_A[1]))
            node_B = (np.int(node_B[0]), np.int(node_B[1]))
            cv2.line(centerline, node_A, node_B, (1, 1, 1), thickness=1)
        centerline_list.append(torch.Tensor(centerline).unsqueeze(0))

    return torch.concat(centerline_list, dim=0)


def refine_graph(graph, image_size, max_connect_dist=30, edge_ignore_size=20):
    """
    Refine graph via connect two endpoints that are adjacent

    Notes:
         graph: networkx.Graph
         image_size: (height, width)

    Args:
        max_connect_dist: max distance for two endpoint to match and connect
        edge_ignore_size: we have to ignore endpoint on the edge, this is the ignore width
    """
    height, width = image_size

    whole_match_dict = {}
    point_to_connect_list = []
    for node, degree in graph.degree():
        if degree == 1 and node[0] > edge_ignore_size and node[1] > edge_ignore_size and \
                node[0] < width - edge_ignore_size and node[1] < height - edge_ignore_size:
            point_to_connect_list.append(node)                  

    if len(point_to_connect_list) > 1:
        point_connect_tree = cKDTree(point_to_connect_list)
        candidate_pairs = point_connect_tree.query_ball_tree(point_connect_tree, r=max_connect_dist)
        done_set = set()

        current_point_idx = 0
        for connect_list in candidate_pairs:
            point_to_connect = list(set(connect_list).difference(done_set))
            if current_point_idx in point_to_connect:
                if len(point_to_connect) >= 2:
                    # have to connect points in point_to_connect
                    points = [point_to_connect_list[i] for i in point_to_connect]
                    centroid = tuple(np.round(np.mean(points, axis=0)).astype(int))
                    # find keypoint that is closet to centroid as new center
                    # dist_list = []
                    # for point in points:
                    #     dist = np.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2)
                    #     dist_list.append(dist)
                    # center_idx = np.argmax(np.array(dist_list))
                    # center = points[center_idx]
                    # points.remove(center)
                    # whole_match_dict[center] = points
                    whole_match_dict[centroid] = points
                done_set.update(point_to_connect)
            current_point_idx += 1

        for center_keypoint in whole_match_dict.keys():
            node_to_remove = []
            edge_to_add = []
            # only update those keypoints that have edges in both patches
            graph_keypoints_list = whole_match_dict[center_keypoint]
            # if len(graph_keypoints_list) == 2:
            #     # if need to connect two points, we only add an edge between them
            #     graph.add_edge(graph_keypoints_list[0], graph_keypoints_list[1])
            if len(graph_keypoints_list) >= 2:
                # if need to connect more, we compute their centroid as new keypoint and delete them all
                for keypoint in graph_keypoints_list:
                    for neighbor in copy.deepcopy(graph.neighbors(keypoint)):
                        edge_to_add.append((neighbor, center_keypoint))
                    node_to_remove.append(keypoint)
                if center_keypoint in node_to_remove:
                    node_to_remove.remove(center_keypoint)
                graph.add_edges_from(edge_to_add)
                graph.remove_nodes_from(node_to_remove)
            # if len(graph_keypoints_list) > 0:
            #     for keypoint in graph_keypoints_list:
            #         graph.add_edge(keypoint, center_keypoint)
                
    return graph


def cut_graph_edge_outside_mask(graph, valid_mask):
    """
    Cut graph edge that outside mask and add new node at the edge of mask.

    Notes:
        graph: networkx.Graph
        valid_mask: torch.zeros (crop_size, crop_size)
    """

    # find corners of valid_mask
    rec_vertices, _ = cv2.findContours(np.uint8(valid_mask), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    left_upper, left_lower, right_lower, right_upper = \
        rec_vertices[0][0][0], rec_vertices[0][1][0], \
        rec_vertices[0][2][0], rec_vertices[0][3][0]  # save as (x, y)

    # find intersection of edge and line
    nodes_to_abandon = []
    edges_to_add = []
    for point in graph.nodes():
        if valid_mask[point[1], point[0]] == 0:
            nodes_to_abandon.append(point)
            for neighbor in graph.neighbors(point):
                if valid_mask[neighbor[1], neighbor[0]] > 0:
                    abandon_node, retain_node = point, neighbor
                    intersection_list = get_intersection_of_line_rectangle(
                        abandon_node, retain_node, left_upper, right_lower
                    )
                    if retain_node in intersection_list:
                        intersection_list.remove(retain_node)
                    if len(intersection_list) != 0:
                        edges_to_add.append((tuple(intersection_list[0]), retain_node))
                elif valid_mask[neighbor[1], neighbor[0]] == 0:
                    nodes_to_abandon.append(neighbor)
                    # two intersection, where edge nodes are outside crop_image region
                    intersection_list = get_intersection_of_line_rectangle(
                        point, neighbor, left_upper, right_lower
                    )
                    if len(intersection_list) == 2:
                        edges_to_add.append((tuple(intersection_list[0]), tuple(intersection_list[1])))

    # remove old nodes and add new lines
    graph.remove_nodes_from(nodes_to_abandon)
    graph.add_edges_from(edges_to_add)

    return graph


def match_around_intersection(tree, A, B, buffer_size=20, max_match_distance=15):
    """
    Match the keypoints around patch intersection line

    Notes:
        tree: rtree.Index
        A, B: intersection line (x, y)

    ATTENTION!! : A should be on the left or above B

    Returns:
        patch_map_dict: {point1: pointA, point2: pointA, point3: point B...} record map information
    """
    # TODO: there is overlap between two buffer
    patch_map_dict = {}
    if A[0] == B[0]:
        # vertical intersection line
        # get keypoint on the both side of the line
        keypoint_left = []
        for point in tree.intersection((A[0] - buffer_size, A[1], B[0] + 1, B[1] + 1), objects=True):
            keypoint_left.append((point.bounds[0] - 0.4, point.bounds[2] - 0.4))
        keypoint_left = torch.Tensor(keypoint_left)

        keypoint_right = []
        for point in tree.intersection((A[0] + 1, A[1], B[0] + buffer_size + 1, B[1] + 1), objects=True):
            keypoint_right.append((point.bounds[0] - 0.4, point.bounds[2] - 0.4))
        keypoint_right = torch.Tensor(keypoint_right)

        # then use Hungarian Match to match these keypoints according to their locations
        if len(keypoint_left) == 0 or len(keypoint_right) == 0:
            return {}
        keypoint_left = keypoint_left.unsqueeze(1)
        keypoint_right = keypoint_right.unsqueeze(0)
        distance_matrix = torch.norm(keypoint_left - keypoint_right, dim=-1)
        left_index, right_index = linear_sum_assignment(distance_matrix)
        # find midpoint of two matched keypoints as new keypoint
        keypoint_left = keypoint_left.squeeze(1)
        keypoint_right = keypoint_right.squeeze(0)
        for i in range(len(left_index)):
            if distance_matrix[left_index[i], right_index[i]] <= max_match_distance:
                keypoint_1 = keypoint_left[left_index[i]]
                keypoint_2 = keypoint_right[right_index[i]]
                new_keypoint = torch.div(keypoint_1 + keypoint_2, 2, rounding_mode='trunc')
                # avoid using tensor
                patch_map_dict[(int(keypoint_1[0]), int(keypoint_1[1]))] = (int(new_keypoint[0]), int(new_keypoint[1]))
                patch_map_dict[(int(keypoint_2[0]), int(keypoint_2[1]))] = (int(new_keypoint[0]), int(new_keypoint[1]))

    elif A[1] == B[1]:
        # horizontal intersection line
        # get keypoint on the both side of the line
        keypoint_up = []
        for point in tree.intersection((A[0], A[1] - buffer_size, B[0] + 1, B[1] + 1), objects=True):
            keypoint_up.append((point.bounds[0] - 0.4, point.bounds[2] - 0.4))
        keypoint_up = torch.Tensor(keypoint_up)

        keypoint_down = []
        for point in tree.intersection((A[0], A[1] + 1, B[0] + 1, B[1] + buffer_size + 1), objects=True):
            keypoint_down.append((point.bounds[0] - 0.4, point.bounds[2] - 0.4))
        keypoint_down = torch.Tensor(keypoint_down)

        # then use Hungarian Match to match these keypoints according to their locations
        if len(keypoint_up) == 0 or len(keypoint_down) == 0:
            return {}
        keypoint_up = keypoint_up.unsqueeze(1)
        keypoint_down = keypoint_down.unsqueeze(0)

        distance_matrix = torch.norm(keypoint_up - keypoint_down, dim=-1)
        up_index, down_index = linear_sum_assignment(distance_matrix)

        # find midpoint of two matched keypoints as new keypoint
        keypoint_up = keypoint_up.squeeze(1)
        keypoint_down = keypoint_down.squeeze(0)
        for i in range(len(up_index)):
            if distance_matrix[up_index[i], down_index[i]] <= max_match_distance:
                keypoint_1 = keypoint_up[up_index[i]]
                keypoint_2 = keypoint_down[down_index[i]]
                new_keypoint = torch.div(keypoint_1 + keypoint_2, 2, rounding_mode='trunc')
                # avoid using tensor
                patch_map_dict[(int(keypoint_1[0]), int(keypoint_1[1]))] = (int(new_keypoint[0]), int(new_keypoint[1]))
                patch_map_dict[(int(keypoint_2[0]), int(keypoint_2[1]))] = (int(new_keypoint[0]), int(new_keypoint[1]))

    return patch_map_dict


def simplify_graph_rdp(graph, rdp_dist=5):
    """ Simplify graph using rdp """
    new_graph = nx.Graph()

    copied_graph = graph.copy()
    intermediates = [node for node, degree in copied_graph.degree() if degree == 2]
    for node in intermediates:
        try:
            copied_graph.add_edge(*copied_graph.neighbors(node))
            copied_graph.remove_node(node)
        except:
            continue

    junctions = [node for node, degree in graph.degree() if degree != 2]
    junctions_set = set(junctions)
    valid_paths = []
    cycle_paths = {}

    for start_node, end_node in copied_graph.edges():
        if start_node in junctions and end_node in junctions:
            copied_graph = graph.copy()
            junction_to_remove = copy.deepcopy(junctions)
            junction_to_remove.remove(start_node)
            if start_node != end_node:
                junction_to_remove.remove(end_node)
            copied_graph.remove_nodes_from(junction_to_remove)

            for path in nx.all_simple_paths(copied_graph, start_node, end_node):
                exclude_set = junctions_set.copy()
                exclude_set.discard(start_node)
                exclude_set.discard(end_node)
                path_set = set(path)
                if exclude_set.isdisjoint(path_set) and len(path_set) >= 2:
                    # do not have other junctions
                    valid_paths.append(path)

    for cycle in nx.cycle_basis(graph):
        cycle_set = set(cycle)
        intersect = cycle_set.intersection(junctions_set)
        cycle = tuple(cycle)

        if len(intersect) == 1:
            cycle_paths[cycle] = intersect.pop()
        elif len(intersect) == 0:
            cycle_paths[cycle] = None

    # rasterize valid_paths and use rdp to simplify it
    for path in valid_paths:
        path_pixels = get_line(path)
        simplified_path = rdp(path_pixels, rdp_dist)
        new_graph.add_edges_from(zip(simplified_path[:-1], simplified_path[1:]))

    # simplify cycles and add it to graph
    for cycle in cycle_paths.keys():
        intersection = cycle_paths[cycle]
        cycle = list(cycle)
        if intersection is not None:
            # cycle with only one junction, have to shift index of junction
            junction_index = cycle.index(intersection)
            cycle = cycle[junction_index:] + cycle[:junction_index]

        mid_index = len(cycle) // 2
        cycle.append(tuple(cycle[0]))
        path_1 = cycle[:mid_index + 1]
        path_2 = cycle[mid_index:]

        path_pixels = get_line(path_1)
        simplified_path = rdp(path_pixels, rdp_dist)
        new_graph.add_edges_from(zip(simplified_path[:-1], simplified_path[1:]))
        path_pixels = get_line(path_2)
        simplified_path = rdp(path_pixels, rdp_dist)
        new_graph.add_edges_from(zip(simplified_path[:-1], simplified_path[1:]))

    return new_graph


def get_line(point_list):
    """
    get line in point_list
    Args:
        point_list [pt1, pt2, ...] pt (row, column)

    Returns:
        [pt1, pt2, ...] pt (column, row)
    """
    valid_pixel_list = []
    for i in range(len(point_list) - 1):
        node_A = point_list[i]
        node_B = point_list[i + 1]
        row, col = draw.line(node_A[0], node_A[1], node_B[0], node_B[1])
        line_pixel_list = [(y, x) for y, x in zip(row, col)]
        line_pixel_list.pop(-1)
        valid_pixel_list += line_pixel_list

    valid_pixel_list.append(point_list[-1])
    return valid_pixel_list


def get_intersection_of_line_rectangle(
        line_node_a, line_node_b, rectangle_left_upper, rectangle_right_lower):
    """
    Find intersection between a line segment and a axis aligned rectangle
    Just use opencv to find the intersection !

    Attention: Node should be (x, y)
    """
    line = sympy.Segment(Point(line_node_a), Point(line_node_b))

    rectangle_right_upper = (rectangle_right_lower[0], rectangle_left_upper[1])
    rectangle_left_lower = (rectangle_left_upper[0], rectangle_right_lower[1])

    rectangle = sympy.Polygon(
        Point(rectangle_left_upper), Point(rectangle_right_upper),
        Point(rectangle_right_lower), Point(rectangle_left_lower)
    )
    intersection_list = rectangle.intersection(line)

    if len(intersection_list) > 0:
        if isinstance(intersection_list[0], sympy.Segment2D):
            intersection_list = [intersection_list[0].points[0], intersection_list[0].points[1]]
    for i in range(len(intersection_list)):
        intersection_list[i] = np.round(np.array(intersection_list[i], dtype=np.float64))
        intersection_list[i] = (int(intersection_list[i][0]), int(intersection_list[i][1]))
    return intersection_list
