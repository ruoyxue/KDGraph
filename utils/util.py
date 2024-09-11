import copy

import cv2
import numpy as np
import torch


def keypoint_visualizer(image, location, direction=None, segmentation=None, mode="pred"):
    """
    Use location, direction to draw keypoints on image

    Notes:
        location: (max_keypoint, 2)    (-1, -1) acts as a placeholder
        direction: (max_keypoint, 6, 3)    probability should be 0 if not exist

    Args:
        image: input image
        location: location of keypoints
        direction: direction of keypoints

    Returns:
        image with keypoint results
    """

    max_keypoint = location.shape[0]
    output_image = copy.deepcopy(image)
  
    location = location.int().cpu()
    location = np.round(np.array(location))

    # direction normalise
    if direction is not None:
        direction = direction.double()
        length = torch.clamp(torch.sqrt(direction[:, :, 1] ** 2 + direction[:, :, 2] ** 2), 1e-8)
        direction[:, :, 1] /= length
        direction[:, :, 2] /= length
        direction = direction.cpu()
        direction = np.array(direction)

    if mode == "pred":
        color_line = (48, 126, 241)
        color_point = (75, 238, 251)
    elif mode == "gt":
        color_line = (151, 85, 47)
        color_point = (240, 176, 0)
    else:
        raise ValueError("Invalid mode in visualizer")

    for i in range(max_keypoint):
        x, y = location[i, 0], location[i, 1]
        if x >= 0 and y >= 0:
            if direction is not None:
                for j in range(6):
                    if direction[i, j, 0] > 0:
                        x_d = np.int(np.round(x + direction[i, j, 2] * 20))
                        y_d = np.int(np.round(y + direction[i, j, 1] * 20))
                        cv2.arrowedLine(output_image, pt1=(x, y), pt2=(x_d, y_d), color=color_line, thickness=3, tipLength=0.1)

    for i in range(max_keypoint):
        x, y = location[i, 0], location[i, 1]
        if x >= 0 and y >= 0:
            cv2.circle(output_image, (x, y), radius=4, color=color_point, thickness=-1)
    return output_image


def vectorization_visualizer_for_graph(image, graph, segmentation=None):
    """
    Use networkx.Graph to draw vectorization output on image

    Args:
        image: image to draw on
        graph: networkx.Graph() of the image
        segmentation: segmentation gt of image

    Returns:
        image with vectorization results
    """
    output_image = copy.deepcopy(image)

    for node_1, node_2 in graph.edges():
        cv2.line(output_image, node_1, node_2, color=(48, 126, 241), thickness=3)

    for node in graph.nodes():
        cv2.circle(output_image, node, radius=4, color=(75, 238, 251), thickness=-1)
    return output_image
