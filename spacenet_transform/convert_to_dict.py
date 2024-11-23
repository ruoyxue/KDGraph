import os
import json
import os
import cv2
import numpy as np
import torch
import rasterio
from decoder import Decoder
from tqdm import tqdm
import networkx as nx
import pickle


def convert_to_dict(json_path, image_path, save_dir, max_keypoint=20000):
    """ convert spacenet dataset to pickle dict for apls evaluation, point save as (x, y) namely (col, row) """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
        image_info_list = json_info["dataset_info"]

    with tqdm(total=len(image_info_list)) as pbar:
        valid_image_name_list = os.listdir(image_path)
        for image_info in image_info_list:
            image_name = image_info["image_name"]
            if image_name in valid_image_name_list:
                # the image info has corresponding image in image_path directory
                location = torch.full((max_keypoint, 2), -1)
                direction = torch.zeros(max_keypoint, 6, 3)
                keypoints = image_info["keypoints"]
                
                count = 0
                for keypoint in keypoints:
                    location[count, :] = torch.Tensor(keypoint["location"])
                    dir_count = 0
                    keypoint_dirs = keypoint["directions"]
                    for i in range(0, len(keypoint_dirs), 2):
                        if (keypoint_dirs[i] is not None) and (keypoint_dirs[i + 1] is not None):
                            direction[count, dir_count, 0] = 1
                            direction[count, dir_count, 1:] = torch.Tensor(keypoint_dirs[i: i + 2])
                        dir_count += 1
                    count += 1

                adjacency_matrix = Decoder.vectorization_decode(
                    location=location.unsqueeze(0), direction=direction.unsqueeze(0),
                    distance_range=[2, 3], point_line_distance=[2, 3]
                ).squeeze(0)

                graph = nx.Graph()
                row, col = torch.where(adjacency_matrix > 0)
                valid_link_list = [(r, c) for r, c, in zip(row, col)]
                for i, j in valid_link_list:
                    if i < j:
                        # cannot use tensor here !!!
                        point1_x = int(location[i][0])
                        point1_y = int(location[i][1])
                        point2_x = int(location[j][0])
                        point2_y = int(location[j][1])
                        graph.add_edge(
                            (point1_x, point1_y), (point2_x, point2_y)
                        )

                pickle_graph = dict()
                for node in graph.nodes():
                    neighbor = []
                    for n in graph.neighbors(node):
                        neighbor.append(n)
                    pickle_graph[node] = neighbor
  
                with open(os.path.join(save_dir, image_name.split(".")[0] + ".pickle"), "wb") as file:
                    pickle.dump(pickle_graph, file)
                
                pbar.update()


def get_original_graph(geojson_path, image_path):
    """
    Get original networkx graph using geojson roads (x, y) (lon, lat)

    Args:
        geojson_path: path of a geojson file
        image_path: path of corresponding image file
    """
    origin_graph = nx.Graph()
    print("Start to get original graph")
    with open(os.path.join(geojson_path), 'r', encoding='utf-8') as geojson_file:
        geojson_info = json.load(geojson_file)
    rasterio_image = rasterio.open(image_path)
    height, width = rasterio_image.read(1).shape

    road_info = geojson_info["features"]
    with tqdm(total=len(road_info)) as pbar:
        for road in road_info:
            # convert geo coords to pixel coords and shift as (x, y)
            data_type = road["geometry"]["type"]
            geo_coord_list = road["geometry"]["coordinates"]

            if data_type == "LineString":
                pixel_coord_list = []
                for lon, lat in geo_coord_list:
                    pixel_coord = rasterio_image.index(lon, lat)
                    pixel_coord_list.append((pixel_coord[1], pixel_coord[0]))

                origin_graph.add_edges_from(zip(pixel_coord_list[:-1], pixel_coord_list[1:]))
            else:
                raise KeyError(f"Find unexpected data type ({data_type})")
            pbar.update()

    return origin_graph


def convert_to_dict_geojson(geojson_path, image_path, save_dir):
    """ convert geojson dataset to pickle dict for apls evaluation, point save as (x, y) namely (col, row) """
    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    graph = get_original_graph(geojson_path=geojson_path, image_path=image_path)

    pickle_graph = dict()
    for node in graph.nodes():
        neighbor = []
        for n in graph.neighbors(node):
            neighbor.append(n)
        pickle_graph[node] = neighbor

    with open(os.path.join(save_dir, image_name.split(".")[0] + ".pickle"), "wb") as file:
        pickle.dump(pickle_graph, file)


if __name__ == "__main__":
    convert_to_dict(
        json_path="/data/spacenet/spacenet.json",
        image_path="/data/spacenet/test/image",
        save_dir="/data/spacenet/test/gt/gt_graph"
    )
   