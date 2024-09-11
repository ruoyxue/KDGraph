import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorLoss(nn.Module):
    def __init__(self, criterion_loc, criterion_dir_prob, criterion_dir_vec,
                 lambda_loc=1, lambda_dir_prob=1, lambda_dir_vec=1):
        super().__init__()
        self.criterion_loc = self.criterion_select(criterion_loc)
        self.criterion_dir_prob = self.criterion_select(criterion_dir_prob)
        self.criterion_dir_vec = self.criterion_select(criterion_dir_vec)
        self.lambda_loc = lambda_loc
        self.lambda_dir_prob = lambda_dir_prob
        self.lambda_dir_vec = lambda_dir_vec

    def criterion_select(self, criterion_name):
        """ select criterion according to name """
        if criterion_name == "weighted_bceloss":
            return Weighted_BCELossWithSigmoid(reduction="mean", smooth=0.001)
        elif criterion_name == "weighted_mseloss":
            return Weighted_MSELoss()
        elif criterion_name == "weighted_smoothl1loss":
            return Weighted_SmoothL1Loss()
        elif criterion_name == "weighted_modifiedfocalloss":
            return Weighted_ModifiedFocalLossWithSigmoid(reduction="mean", alpha=2, beta=4, smooth=0.001)
        else:
            raise ValueError(f"Wrong Criterion Type ({criterion_name})")

    def forward(self, pred_info, gt_info, valid_masks, epoch=0):
        """
        Notes:
            pred_info: (pred_location_feat, pred_direction_feat)  (batch, channel, height, width)
            gt_info: (gt_location_feat, gt_direction_feat)
            valid_masks: (batch, 1, height, width)
        """
        pred_location_feat, pred_direction_feat = pred_info
        gt_location_feat, gt_direction_feat = gt_info
        assert pred_location_feat.shape[0] == gt_location_feat.shape[0]

        weight = copy.deepcopy(gt_location_feat)
        
        weight[weight == 0] = 0.002  # crop_size 448 radius 3

        loss_loc = self.criterion_loc(pred_location_feat, gt_location_feat, weight=weight, valid_masks=valid_masks)
        pred_prob = pred_direction_feat[:, [0, 3, 6, 9, 12, 15], :, :]
        gt_prob = gt_direction_feat[:, [0, 3, 6, 9, 12, 15], :, :]
        pred_vec = pred_direction_feat[:, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17], :, :]
        gt_vec = gt_direction_feat[:, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17], :, :]
        
        loss_dir_prob = self.criterion_dir_prob(pred_prob, gt_prob, weight=weight, valid_masks=valid_masks)
        loss_dir_vec = self.criterion_dir_vec(pred_vec, gt_vec, weight=gt_location_feat, valid_masks=valid_masks)
        loss_sum = self.lambda_loc * loss_loc + self.lambda_dir_prob * loss_dir_prob + self.lambda_dir_vec * loss_dir_vec
        return loss_loc, loss_dir_prob, loss_dir_vec, loss_sum


def to_smoothed_one_hot(in_tensor, n_class, eps=0.):
    """ Convert pixel-level segmentation gt to one-hot code, together with label smooth
    Notes:
        in_tensor: (batch, 1, height, width)

    Args:
        eps: true class -> 1 - eps, other classes -> eps / (n_classes - 1)

    Returns:
        one_hot: (batch, n_class, height, width)
    """
    assert len(in_tensor.shape) == 4
    in_tensor = in_tensor.long()
    batch_size, _, height, width = in_tensor.shape
    one_hot = torch.zeros(batch_size, n_class, height, width).to(in_tensor.device)
    one_hot = one_hot.scatter_(1, in_tensor, 1)

    if eps != 0:
        mask = ~(one_hot > 0)
        one_hot = torch.masked_fill(one_hot, mask, eps / (n_class - 1))
        one_hot = torch.masked_fill(one_hot, ~mask, 1 - eps)
    return one_hot


class Weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, weight, valid_masks=None):
        """
        Notes:
            pred: (batch, channel, height, width)
            gt: (batch, channel, height, width)
            weight: (batch, 1, height, width)
            valid_masks: (batch, 1, height, width)
        """
        assert pred.numel() == gt.numel()
        tem = torch.sum(torch.pow(pred - gt, 2), dim=1) * valid_masks

        loss = torch.sum(tem * weight * valid_masks) / torch.clamp(torch.sum(weight * valid_masks), min=1e-8)
        return torch.clamp(loss, max=100)


class Weighted_SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, weight):
        """
        Notes:
            pred: (batch, channel, height, width)
            gt: (batch, channel, height, width)
            weight: (batch, 1, height, width)
        """
        loss = F.smooth_l1_loss(pred * weight, gt * weight, reduction='sum')
        loss = loss / torch.clamp(torch.sum(weight), min=1e-8)
        return loss


class Weighted_BCELossWithSigmoid(nn.Module):
    """ Just for Foreground 1 and Background 0 Segmentation """
    def __init__(self, reduction='mean', smooth=0.):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        assert 0 <= smooth < 1, "smooth value should be in [0,1]"
        self.smooth = smooth

    def forward(self, pred, gt, weight=None, valid_masks=None):
        """
        Notes:
            pred, gt, weight: (batch, 1, height, width)
        """

        assert pred.shape[0] == gt.shape[0]

        batch_size = pred.shape[0]
        pred_foreground = torch.log(torch.clamp(torch.sigmoid(pred), min=1e-8))
        pred_background = torch.log(torch.clamp(1 - torch.sigmoid(pred), min=1e-8))

        # one_hot_gt = to_smoothed_one_hot(gt, n_class=2, eps=self.smooth)
        loss = - ((pred_foreground * gt) + pred_background * (1 - gt))

        if weight is not None:
            loss *= weight
        if valid_masks is not None:
            loss *= valid_masks
        
        if self.reduction == 'none':
            return loss.view(batch_size, -1).mean(1)
        elif self.reduction == 'mean':
            if weight is not None and valid_masks is not None:
                return torch.clamp(torch.sum(loss) / torch.clamp(torch.sum(weight * valid_masks), min=1e-8), max=100)
            elif weight is None and valid_masks is not None:
                return torch.clamp(torch.sum(loss) / torch.clamp(torch.sum(valid_masks), min=1e-8), max=100)
            elif weight is not None and valid_masks is None:
                return torch.clamp(torch.sum(loss) / torch.clamp(torch.sum(weight), min=1e-8), max=100)
            else:
                return torch.clamp(torch.mean(loss) * 10, max=100)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")


class Weighted_ModifiedFocalLossWithSigmoid(nn.Module):
    """ modified focal loss in Cornernet, with small modification """
    def __init__(self, reduction='mean', smooth=0., alpha=2, beta=4):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        assert 0 <= smooth < 1, "smooth value should be in [0,1]"
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, weight=None, valid_masks=None):
        """
        Notes:
            pred, gt, weight: (batch, 1, height, width)
        """
        assert pred.shape[0] == gt.shape[0]
        batch_size = pred.shape[0]
        gt_copy = copy.deepcopy(gt)
        gt_copy = torch.clamp(gt_copy - 2 * self.smooth, min=0)
        gt_copy += self.smooth

        pred = torch.clamp(torch.sigmoid(pred), min=1e-8)
        # positive_loss = torch.pow(torch.abs(gt_copy - pred), self.alpha) * torch.pow(gt_copy, self.beta) * torch.log(pred)
        positive_loss = torch.pow(1 - pred, self.alpha) * torch.pow(gt_copy, self.beta) * torch.log(pred)
        negative_loss = torch.pow(pred, self.alpha) * torch.pow(1 - gt_copy, self.beta) * torch.log(torch.clamp(1 - pred, min=1e-8))
        if weight is not None:
            loss = - weight * (positive_loss + negative_loss)
        else:
            loss = - (positive_loss + negative_loss)

        if valid_masks is not None:
            loss *= valid_masks

        if self.reduction == 'none':
            return loss.view(batch_size, -1).mean(1)
        elif self.reduction == 'mean':
            if weight is not None and valid_masks is not None:
                return torch.clamp(torch.sum(loss) / torch.clamp(torch.sum(weight * valid_masks), min=1e-8), max=100)
            elif weight is None and valid_masks is not None:
                return torch.clamp(torch.sum(loss) / torch.clamp(torch.sum(valid_masks), min=1e-8), max=100)
            elif weight is not None and valid_masks is None:
                return torch.clamp(torch.sum(loss) / torch.clamp(torch.sum(weight), min=1e-8), max=100)
            else:
                return torch.clamp(torch.mean(loss) * 10, max=100)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")


class BinaryDiceLossWithSigmoid(nn.Module):
    def __init__(self, reduction='mean', smooth=0.):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction
        assert 0 <= smooth < 1, "smooth value should be in [0,1]"
        self.smooth = smooth

    def forward(self, pred, gt):
        """
        Notes:
            pred, gt: (batch, 1, height, width)
        """
        assert pred.shape[0] == gt.shape[0]
        gt_copy = copy.deepcopy(gt)
        gt_copy = torch.clamp(gt_copy - 2 * self.smooth, min=0)
        gt_copy += self.smooth
        pred = torch.sigmoid(pred)

        gt_flatten = gt_copy.flatten(start_dim=1)  # (batch, height * width)
        pred_flatten = pred.flatten(start_dim=1)  # (batch, height * width)

        intersection = gt_flatten * pred_flatten

        loss = (2 * intersection.sum(1)) / torch.clamp(gt_flatten.sum(1) + pred_flatten.sum(1), min=1e-8)
        loss = 1 - loss

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class BinaryComposedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = Weighted_BCELossWithSigmoid()
        self.dice = BinaryDiceLossWithSigmoid()

    def forward(self, pred, gt):
        """
        Notes:
            pred: (batch, 1, height, width)
            gt: (batch, 1, height, width)
        """
        return self.bce(pred, gt) + self.dice(pred, gt)
