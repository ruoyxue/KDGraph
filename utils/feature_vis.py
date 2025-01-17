import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os


def min_max_normalise(x):
    """
    use min-max normalisation to normalise the tensor

    Notes:
        x: (1, 1, height, width) torch.tensor
    """
    x = x - torch.min(x) / torch.clamp(torch.max(x) - torch.min(x), min=1e-8)
    return x


class FeatureVisualizer:
    """
    Visualize feature map and merge it with image

    Notes:
        model: nn.Module
        target_layer_list: [str]
    """
    def __init__(self, model, target_layer_list=None):
        self.model = model
        self.target_layer_list = target_layer_list

    def get_target_output(self, x):
        raise NotImplementedError

    def apply_heatmap_on_image(self, image, heatmap, apply_feat_on_image=True):
        """
        Draw heatmap on image

        Notes:
            image: (height, width, 3)
            heatmap: (height, width)
        """
        heatmap = cv2.merge([heatmap, heatmap, heatmap])  # (height, width, 3)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        if heatmap.shape != image.shape:
            heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
        if apply_feat_on_image:
            output_image = cv2.addWeighted(image, 0.2, heatmap, 0.8, gamma=0)
        else:
            output_image = heatmap
        return output_image

    def save_feature(self, image, target_output_dict, image_name, save_dir, apply_feat_on_image=True):
        """
        Notes:
            image: (height, width, 3)
            gt_locs: (max_keypoint, 2)
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for key in target_output_dict.keys():
            heatmap = target_output_dict[key]
            output_image = self.apply_heatmap_on_image(image=image, heatmap=heatmap, apply_feat_on_image=apply_feat_on_image)
            save_name = image_name.split(".")[0] + "_" + key + ".png"
            cv2.imwrite(os.path.join(save_dir, save_name), output_image)

    def get_cam_intermediate_feature(self, image, feature):
        """
        Get cam of a intermediate feature tensor

        Notes:
            feature: (1, channel, feature_height, feature_width)
            image: (1, 3, height, width)

        Returns:
            cam: (height, width) np.uint8
            min_value: min value of cam
            max_value: max value of cam
        """
        assert image.shape[0] == feature.shape[0] == 1
        channel = feature.shape[1]
        _, _, height, width = image.shape

        cam = torch.ones((1, 1, height, width), dtype=torch.float32)
        for i in range(channel):
            saliency_map = feature[:, i, :, :].unsqueeze(1)  # (1, 1, feature_height, feature_width)
            saliency_map = F.interpolate(saliency_map, size=(height, width), mode='bilinear', align_corners=False)
            saliency_map = min_max_normalise(saliency_map)  # (1, 1, height, width)
            f_loc, _ = self.model(saliency_map * image)
            cam += torch.sigmoid(f_loc) * saliency_map

        cam = cam / channel
        min_value = torch.min(cam)
        max_value = torch.max(cam)
        cam = min_max_normalise(cam)
        cam = (cam.cpu().squeeze().detach().numpy() * 255).astype(np.uint8)
        return cam, min_value, max_value

    def get_cam_output_heatmap(self, feature, sigmoid=True, verbose=False):
        """
        Get cam of output heatmap

        Notes:
            feature: (1, channel, height, width)

        Returns:
            cam: (height, width)
        """
        feature = feature.cpu()
        if verbose:
            print("\nmax:", torch.max(feature))
            print("min:", torch.min(feature))

        if sigmoid:
            _, channel, height, width = feature.shape
            if channel > 1:
                tem = torch.zeros((height, width))
                for i in range(channel):
                    tem += torch.sigmoid(feature[0, i, :, :])
                cam = tem / channel
            else:
                cam = torch.sigmoid(feature)
        else:
            cam = feature
        cam = (cam.squeeze().detach().numpy() * 255).astype(np.uint8)
        return cam


class KDGraph_Feature_Visualizer(FeatureVisualizer):
    def __init__(self, model):
        super().__init__(model)

    def get_target_output(self, x):
        """
        Forward pass model and get outputs of target layer

        Notes:
            x: (1, 3, height, width)
            Every target layer output would be ()
        """
        target_output_dict = {}
        loc_feat, dir_feat = self.model(x)

        target_output_dict["loc_feat"] = self.get_cam_output_heatmap(feature=loc_feat)
        for i in range(0, 18, 3):
            target_output_dict[f"dir_prob_feat_{i}"] = self.get_cam_output_heatmap(feature=dir_feat[:, i, :, :].unsqueeze(1))
        # target_output_dict["final"] = self.get_cam_output_heatmap(feature=loc_feat+dir_prob_feat)
        
        return target_output_dict
