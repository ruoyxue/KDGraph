import copy
import networkx as nx
import sympy
import json
import os
from tqdm import tqdm
import cv2
import numpy as np
from scipy.spatial import cKDTree
from sympy import Point, Line, Segment
from spacenet_transform import get_original_graph, simplify_graph, get_keypoint_info


def generate_cropped_data(image, graph, crop_size=448, overlap_size=0):
    """
    Crop image to small-size images with overlap

    Args:
        image: (height, width, 3) .tif
        graph: simplified networkx.Graph (x, y)
        crop_size: the size of cropped images
        overlap_size: the size of overlap

    Returns:
        cropped_image_list, cropped_graph_list
    """
    height, width = image.shape[:2]
    cropped_image_list, cropped_graph_list = [], []

    # (x, y) is the left-upper coordinate of the image patch
    x_list = list(range(0, width - crop_size, crop_size - overlap_size)) + [width - crop_size]
    y_list = list(range(0, height - crop_size, crop_size - overlap_size)) + [height - crop_size]

    for x in x_list:
        for y in y_list:
            # crop_image = image[y - crop_size + 1: y + 1, x - crop_size + 1: x + 1, :]
            crop_image = copy.deepcopy(image[y: y + crop_size, x: x + crop_size, :])
            graph_copy = copy.deepcopy(graph)
            abandon_dict = {}
            for node in graph_copy.nodes():
                abandon_dict[node] = False

            # invalid mask tells which nodes should be abandoned
            invalid_mask = np.ones((height, width), dtype=np.uint8)
            invalid_mask[y: y + crop_size, x: x + crop_size] = 0
            for node in graph_copy.nodes():
                if invalid_mask[node[1], node[0]]:
                    abandon_dict[node] = True

            # find intersections on the crop_image boundary, and add edges
            edges_to_add = []
            for edge in graph_copy.edges():
                if (abandon_dict[edge[0]] is True) and (abandon_dict[edge[1]] is False):
                    # only one intersection, one inside the other outside
                    abandon_node, retain_node = edge[0], edge[1]
                    intersection_list = get_intersection_of_line_rectangle(
                        abandon_node, retain_node, (x, y), (x + crop_size - 1, y + crop_size - 1)
                    )
                    if retain_node in intersection_list:
                        intersection_list.remove(retain_node)
                    if len(intersection_list) != 0:
                        edges_to_add.append((tuple(intersection_list[0]), retain_node))
                elif (abandon_dict[edge[0]] is False) and (abandon_dict[edge[1]] is True):
                    # only one intersection, one inside the other outside
                    abandon_node, retain_node = edge[1], edge[0]
                    intersection_list = get_intersection_of_line_rectangle(
                        abandon_node, retain_node, (x, y), (x + crop_size - 1, y + crop_size - 1)
                    )
                    if retain_node in intersection_list:
                        intersection_list.remove(retain_node)
                    if len(intersection_list) != 0:
                        edges_to_add.append((tuple(intersection_list[0]), retain_node))
                elif (abandon_dict[edge[0]] is True) and (abandon_dict[edge[1]] is True):
                    # two intersection, where edge nodes are outside crop_image region
                    if (edge[0][0] < x and edge[1][0] < x) or (edge[0][0] > x + crop_size and edge[1][0] > x + crop_size) or \
                            (edge[0][1] < y and edge[1][1] < y) or (edge[0][1] > y + crop_size and edge[1][1] > y + crop_size):
                        continue
                    intersection_list = get_intersection_of_line_rectangle(
                        edge[0], edge[1], (x, y), (x + crop_size - 1, y + crop_size - 1)
                    )
                    if len(intersection_list) == 2:
                        edges_to_add.append((tuple(intersection_list[0]), tuple(intersection_list[1])))

            # remove nodes that should be abandoned
            nodes_to_abandon = []
            for node in graph_copy.nodes():
                if abandon_dict[node] is True:
                    nodes_to_abandon.append(node)

            graph_copy.remove_nodes_from(nodes_to_abandon)
            graph_copy.add_edges_from(edges_to_add)

            # shift location of other nodes
            new_edges = []
            for node_1, node_2 in graph_copy.edges():
                new_node1 = (node_1[0] - x, node_1[1] - y)
                new_node2 = (node_2[0] - x, node_2[1] - y)
                new_edges.append((new_node1, new_node2))

            crop_graph = nx.Graph()
            crop_graph.add_edges_from(new_edges)

            for _ in range(2):
                for _ in range(2):
                    crop_graph = merge_dense_keypoints(crop_graph, merge_dist=20)

                for _ in range(3):
                    crop_graph = separate_keypoints_near_edge(
                        crop_graph, (crop_size, crop_size), edge_width=15, min_dist=15)

            cropped_image_list.append(crop_image)
            cropped_graph_list.append(crop_graph)

    return cropped_image_list, cropped_graph_list


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
        if center_keypoint in node_to_remove:
            node_to_remove.remove(center_keypoint)
        graph.add_edges_from(edge_to_add)
        graph.remove_nodes_from(node_to_remove)

    # remove isolated nodes
    nodes = copy.deepcopy(graph.nodes())
    for node in nodes:
        if len(list(graph.neighbors(node))) == 0:
            graph.remove_node(node)

    return graph


def separate_keypoints_near_edge(graph, image_shape, edge_width=10, min_dist=10):
    """
    Separate endpoints near the edge, to make these points not close to each other

    Args:
        graph (networkx.Graph): input graph
        image_shape (Tuple(int, int)): image shape
        edge_width: (float): processing area width
        min_dist (float): min distance of two endpoints

    Returns:
        refined graph
    """
    height, width = image_shape

    pending_point_list = []
    result_list = []  # record pairs of points that are close to each other
    for node, degree in graph.degree():
        if degree == 1 and not (node[0] > edge_width and node[1] > edge_width and \
           node[0] < width - edge_width and node[1] < height - edge_width):
            pending_point_list.append(node)

    if len(pending_point_list) > 1:
        point_connect_tree = cKDTree(pending_point_list)
        candidate_pairs = point_connect_tree.query_ball_tree(point_connect_tree, r=min_dist)
        done_set = set()

        current_point_idx = 0
        for connect_list in candidate_pairs:
            point_to_connect = list(set(connect_list).difference(done_set))
            if current_point_idx in point_to_connect:
                if len(point_to_connect) >= 2:
                    points = [pending_point_list[i] for i in point_to_connect]
                    result_list.append(points)
                done_set.update(point_to_connect)
            current_point_idx += 1

        # random move points until satisfy the min distance request
        match_dict = {}
        for pairs in result_list:
            dist1 = -1
            dist2 = -1
            count = 0
            while dist1 < min_dist or dist2 > 1.8 * min_dist:
                count += 1
                if count > 100:
                    new_point_list = pairs
                    break
                new_point_list = []
                for i in range(len(pairs)):
                    point_x, point_y = pairs[i]
                    move_x, move_y = np.random.randint(low=-int(min_dist / 2) - 1, high=int(min_dist / 2) + 2, size=2)
                    
                    while((point_x + move_x < 0) or (point_x + move_x >= width)
                          or (point_y + move_y < 0) or (point_y + move_y >= height)):
                        # assure that random move is valid
                        
                        move_x, move_y = np.random.randint(low=-int(min_dist / 2)-1, high=int(min_dist / 2) + 2, size=2)
                    new_point_list.append((point_x + move_x, point_y + move_y))

                # calculate if new points satisfy the request
                dist_list = []
                for i in range(len(new_point_list)):
                    for j in range(i + 1, len(new_point_list)):
                        dist_list.append(node_dist(new_point_list[i], new_point_list[j]))
                dist1 = min(dist_list)
                dist2 = max(dist_list)

            # save move info in dict
            for i in range(len(new_point_list)):
                match_dict[pairs[i]] = new_point_list[i]

        for old_keypoint, new_keypoint in match_dict.items():
            if old_keypoint != new_keypoint:
                node_to_remove = []
                edge_to_add = []
                for neighbor in copy.deepcopy(graph.neighbors(old_keypoint)):
                    edge_to_add.append((neighbor, new_keypoint))
                node_to_remove.append(old_keypoint)

                graph.add_edges_from(edge_to_add)
                graph.remove_nodes_from(node_to_remove)

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


def compute_valid_image_ratio(image):
    """ cumpute valid ratio of image """
    height, width = image.shape[:2]
    invalid_mask = (image[:, :, 0] == 5) * (image[:, :, 1] == 5) * (image[:, :, 2] == 5)
    invalid_area = np.sum(invalid_mask)
    return 1 - invalid_area / (height * width)


if __name__ == "__main__":
    image_dir_path_list = [
        r"E:\dataset\spacenet_dataset\AOI_5_Khartoum\PS-RGB-8Bit"
    ]

    geojson_dir_path_list = [
        r"E:\spacenet\Paris\geojson_roads"
    ]

    rdp_dist = 2
    crop_size = 448
    overlap_size = 0
    save_path = os.path.join("E:/spacenet", f"CROP_{crop_size}_OVERLAP_{overlap_size}")

    save_image_path = os.path.join(save_path, "image")
    save_mask_path = os.path.join(save_path, "gt")
    save_json_path = os.path.join(save_path, "whole.json")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_mask_path, exist_ok=True)

    assert len(image_dir_path_list) == len(geojson_dir_path_list)
    image_count = 0
    for image_dir in image_dir_path_list:
        image_count += len(os.listdir(image_dir))
    dataset_infos = []
    dir_len = len(image_dir_path_list)
    sum_count = 0

    with tqdm(total=image_count, unit_scale=True, unit=" file", ncols=100) as pbar:
        for directory_count in range(dir_len):
            current_image_dir_path = image_dir_path_list[directory_count]
            current_geojson_dir_path = geojson_dir_path_list[directory_count]

            for image_name in os.listdir(current_image_dir_path):
                geojson_name = image_name.replace("PS-RGB", "geojson_roads").replace("tif", "geojson")
                if not os.path.exists(os.path.join(current_geojson_dir_path, geojson_name)):
                    continue
                origin_graph = get_original_graph(
                    geojson_path=os.path.join(current_geojson_dir_path, geojson_name),
                    image_path=os.path.join(current_image_dir_path, image_name)
                )

                image = cv2.imread(os.path.join(current_image_dir_path, image_name))
                cropped_image_list, cropped_graph_list = generate_cropped_data(
                    image=image, graph=origin_graph, crop_size=crop_size, overlap_size=overlap_size
                )

                count = 0
                for i in range(len(cropped_image_list)):
                    if compute_valid_image_ratio(cropped_image_list[i]) < 0.15:
                        # delete image with small ratio of valid info
                        continue

                    simplified_graph = simplify_graph(cropped_graph_list[i], rdp_dist=rdp_dist)
                    keypoint_info = get_keypoint_info(simplified_graph)

                    tem_list = []
                    for keypoint in list(keypoint_info.keys()):
                        tem_list.append({
                            "location": [float(keypoint[0]), float(keypoint[1])],
                            "directions": keypoint_info[keypoint]
                        })
                    crop_name = image_name.split(".")[0] + "_" + str(count) + ".png"
                    info = {
                        "image_name": crop_name,
                        "height": crop_size,
                        "width": crop_size,
                        "keypoints": tem_list
                    }
                    dataset_infos.append(info)

                    mask = np.zeros((crop_size, crop_size, 3))
                    for node_1, node_2 in simplified_graph.edges():
                        cv2.line(mask, node_1, node_2, (255, 255, 255), thickness=25)
                    cv2.imwrite(os.path.join(save_mask_path, crop_name), mask)
                    cv2.imwrite(os.path.join(save_image_path, crop_name), cropped_image_list[i])
                    count += 1
                    sum_count += 1
                pbar.update()

    print("sum count: ", sum_count)
    info = {"dataset_info": dataset_infos}
    with open(save_json_path, 'w', encoding='utf-8') as fw:
        json.dump(info, fw, indent=4, ensure_ascii=False)
