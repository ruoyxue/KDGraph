import copy

import networkx as nx
import json
import os
import rasterio
from tqdm import tqdm
from crdp import rdp
import numpy as np
from skimage import draw
import math
from scipy.optimize import linear_sum_assignment


def get_original_graph(geojson_path, image_path):
    """
    Get original networkx graph using geojson roads (x, y)

    Args:
        geojson_path: path of a geojson file
        image_path: path of corresponding image file
    """
    origin_graph = nx.Graph()

    geojson_name = os.path.basename(geojson_path)
    image_name = os.path.basename(image_path)
    assert image_name.replace("PS-RGB", "geojson_roads").replace("tif", "geojson") == geojson_name

    with open(os.path.join(geojson_path), 'r', encoding='utf-8') as geojson_file:
        geojson_info = json.load(geojson_file)
    rasterio_image = rasterio.open(os.path.join(image_path))

    road_info = geojson_info["features"]
    for road in road_info:
        # convert geo coords to pixel coords and shift as (x, y)
        data_type = road["geometry"]["type"]
        geo_coord_list = road["geometry"]["coordinates"]

        if data_type == "LineString":
            pixel_coord_list = []
            for lat, lon in geo_coord_list:
                pixel_coord = rasterio_image.index(lat, lon)
                if pixel_coord[0] == 1300:
                    pixel_coord = (pixel_coord[0] - 1, pixel_coord[1])
                if pixel_coord[1] == 1300:
                    pixel_coord = (pixel_coord[0], pixel_coord[1] - 1)
                pixel_coord_list.append((pixel_coord[1], pixel_coord[0]))
            origin_graph.add_edges_from(zip(pixel_coord_list[:-1], pixel_coord_list[1:]))
        elif data_type == "MultiLineString":
            geo_coord_list = road["geometry"]["coordinates"]
            for single_linestring in geo_coord_list:
                pixel_coord_list = []
                for lat, lon in single_linestring:
                    pixel_coord = rasterio_image.index(lat, lon)
                    if pixel_coord[0] == 1300:
                        pixel_coord = (pixel_coord[0] - 1, pixel_coord[1])
                    if pixel_coord[1] == 1300:
                        pixel_coord = (pixel_coord[0], pixel_coord[1] - 1)
                    pixel_coord_list.append((pixel_coord[1], pixel_coord[0]))
                origin_graph.add_edges_from(zip(pixel_coord_list[:-1], pixel_coord_list[1:]))
        else:
            raise KeyError(f"Find unexpected data type ({data_type})")

    return origin_graph


def simplify_graph(graph, rdp_dist=5):
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
        # print(start_node in junctions)
        # print(end_node in junctions)
        # print(start_node in junctions and end_node in junctions)
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


def get_degree(sin_value, cos_value):
    """ get angle in [0, 360) via sin and cos value """
    theta = math.atan2(sin_value, cos_value)
    if theta < 0:
        theta += 2 * math.pi
    return math.degrees(theta)


def sort_direction(direction, max_direction):
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


def get_keypoint_info(graph):
    """
    Generate keypoint info of an image according to graph dict

    Return:
        keypoint_info_dict: {current_point: [dir1_value1, dir1_value2, dir2_value1, dir2_value2, ...], ...}
    """
    keypoint_info_dict = {}
    for node in graph.nodes:
        dir_list = []
        for neighbor in graph.neighbors(node):
            # for keypoint
            length = np.linalg.norm(np.array(neighbor) - np.array(node))
            dir_list.append((neighbor[1] - node[1]) / length)
            dir_list.append((neighbor[0] - node[0]) / length)
        keypoint_info_dict[node] = sort_direction(dir_list, 6)
    return keypoint_info_dict


if __name__ == "__main__":
    image_dir_path_list = [
        r"E:\spacenet\AOI_2_Vegas\PS-RGB-8Bit"
    ]

    geojson_dir_path_list = [
        r"E:\spacenet\AOI_2_Vegas\geojson_roads"
    ]

    save_path = "./spacenet.json"
    assert len(image_dir_path_list) == len(geojson_dir_path_list)
    rdp_dist = 3

    image_count = 0
    for image_dir in image_dir_path_list:
        image_count += len(os.listdir(image_dir))
    dataset_infos = []
    dir_len = len(image_dir_path_list)
    count = 0

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
                simplified_graph = simplify_graph(origin_graph, rdp_dist=rdp_dist)

                keypoint_info = get_keypoint_info(simplified_graph)

                tem_list = []
                for keypoint in list(keypoint_info.keys()):
                    tem_list.append({
                        "location": [float(keypoint[0]), float(keypoint[1])],
                        "directions": keypoint_info[keypoint]
                    })

                # height, width = cv2.imread(os.path.join(current_image_dir_path, image_name)).shape[:2]
                height, width = (1300, 1300)
                info = {
                    "image_name": image_name,
                    "height": height,
                    "width": width,
                    "keypoints": tem_list
                }
                dataset_infos.append(info)
                count += 1
                pbar.update()

    print("count: ", count)
    info = {"dataset_info": dataset_infos}
    with open(save_path, 'w', encoding='utf-8') as fw:
        json.dump(info, fw, indent=4, ensure_ascii=False)
