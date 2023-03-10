import math
import torch
import torchvision

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops import clip_boxes_to_image
from torchvision.ops.boxes import batched_nms
from torch.nn.functional import one_hot
from torchvision.ops import box_convert

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss
from torch.nn.functional import binary_cross_entropy_with_logits

class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=2, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        
        out = []
        for i in x:
            i = self.conv(i)
            i = self.cls_logits(i)
            out.append(i)
        
        return out


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=2):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        regression = []
        centerness = []
        for i in x:
            iconv = self.conv(i)
            iregress = self.bbox_reg(iconv)
            regression.append(iregress)
            icenter = self.bbox_ctrness(iconv)
            centerness.append(icenter)
            
        return regression, centerness


class FCOS(nn.Module):
    """
    Implementation of (simplified) Fully Convolutional One-Stage object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet18 is supported
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
        devices
    ):
        super().__init__()
        assert backbone == "ResNet18"
        self.backbone_name = backbone
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network (resnet18)
        self.backbone = create_feature_extractor(
            resnet18(weights=ResNet18_Weights.DEFAULT), return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

        self.devices = devices

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
            else:
                module.train(mode)
        return self

    """
    The behavior of the forward function changes depending if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)
        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        BIG_NUMBER = 1e8

        device = self.devices[0]

        positive_samples = 0
        
        cls_loss, reg_loss, ctr_loss = [], [], []
        
        for layer in range(len(cls_logits)):
            cls_logits[layer]   = cls_logits[layer].permute((0,2,3,1))      # N_images x H x W x 20
            reg_outputs[layer]  = reg_outputs[layer].permute((0,2,3,1))     # N_images x H x W x 4
            ctr_logits[layer]   = ctr_logits[layer].permute((0,2,3,1))      # N_images x H x W x 1

        for image, targets_per_image in enumerate(targets):
            target_boxes    = targets_per_image['boxes']                                                                # N_boxes x 4   (x1,y1,x2,y2)
            target_label    = targets_per_image['labels']                                                               # N_boxes x 1   (class in range [1:20])
            target_areas    = (target_boxes[:,2] - target_boxes[:,0]) * (target_boxes[:,3] - target_boxes[:,1])         # N_boxes x 1   (area)
            target_centers  = (target_boxes[:,:2] + target_boxes[:,2:]) / 2                                             # N_boxes x 2   (x,y)

            for layer_points, layer_stride, layer_reg_range, layer_cls_logits, layer_reg_outputs, layer_ctr_logits in \
                zip(points, strides, reg_range, cls_logits, reg_outputs, ctr_logits):

                cls_logits_per_point    = layer_cls_logits[image]       # H x W x 20
                reg_outputs_per_point   = layer_reg_outputs[image]      # H x W x 4
                ctr_logits_per_point    = layer_ctr_logits[image]       # H x W x 1

                W, H = layer_points.shape[:2]       # layer_points - H x W x 2      (x,y)
                N_boxes = target_boxes.shape[0]     # target_boxes - N_boxes x 4    (x1,y1,x2,y2)

                center_boxes_x1y1 = target_centers - self.center_sampling_radius * layer_stride         # N_boxes x 2
                center_boxes_x2y2 = target_centers + self.center_sampling_radius * layer_stride         # N_boxes x 2
                target_center_boxes = torch.concat((center_boxes_x1y1, center_boxes_x2y2), dim = 1)     # N_boxes x 4
                
                repeated_layer_points = layer_points.unsqueeze(dim=0).repeat(N_boxes, 1, 1, 1)          # convert layer_points from H*W*2 to N_boxes*H*W*2
                repeated_target_subboxes = target_center_boxes.view(-1, 1, 1, 4).repeat(1, W, H, 1)     # convert target_center_boxes from N_boxes*4 to N_boxes*H*W*4

                point_x = repeated_layer_points[:,:,:,0]            # N_boxes x H x W
                point_y = repeated_layer_points[:,:,:,1]            # N_boxes x H x W
                
                subbox_x1 = repeated_target_subboxes[:,:,:,0]       # N_boxes x H x W
                subbox_y1 = repeated_target_subboxes[:,:,:,1]       # N_boxes x H x W
                subbox_x2 = repeated_target_subboxes[:,:,:,2]       # N_boxes x H x W
                subbox_y2 = repeated_target_subboxes[:,:,:,3]       # N_boxes x H x W
                
                repeated_target_boxes = target_boxes.view(-1, 1, 1, 4).repeat(1, W, H, 1)  
                target_box_x1 = repeated_target_boxes[:,:,:,0]       # N_boxes x H x W
                target_box_y1 = repeated_target_boxes[:,:,:,1]       # N_boxes x H x W
                target_box_x2 = repeated_target_boxes[:,:,:,2]       # N_boxes x H x W
                target_box_y2 = repeated_target_boxes[:,:,:,3]       # N_boxes x H x W

                l = (point_x - target_box_x1)             # N_boxes x H x W
                t = (point_y - target_box_y1)             # N_boxes x H x W
                r = (target_box_x2 - point_x)             # N_boxes x H x W
                b = (target_box_y2 - point_y)             # N_boxes x H x W
                
                # add extra dimension at the end so that concat on last dim returns tensor of shape (N_boxes x H x W x 4)
                #why ??
                ll = l.unsqueeze(-1)                                 # N_boxes x H x W x 1
                tt = t.unsqueeze(-1)                                 # N_boxes x H x W x 1
                rr = r.unsqueeze(-1)                                 # N_boxes x H x W x 1
                bb = b.unsqueeze(-1)                                 # N_boxes x H x W x 1

                max_dist = torch.max(torch.cat((ll, tt, rr, bb), dim=-1), dim=-1)[0]   # N_boxes x H x W

                # each point must satisfy below conditions:
                # 1. point (x,y) must be within target subbox
                # 2. point (x,y) must be within target box      (required as some certain edge-cases lie within target subbox but outside target box)
                # 3. max(l,t,r,b) for that point must be within layer_reg_range

                mask = torch.where(                     # N_boxes x H x W   (True/False)
                    # point lies in target-subbox
                    (point_x >= subbox_x1) & 
                    (point_x <= subbox_x2) &
                    (point_y >= subbox_y1) & 
                    (point_y <= subbox_y2) & 

                    # point lies in target-box
                    (point_x >= target_box_x1) & 
                    (point_x <= target_box_x2) &
                    (point_y >= target_box_y1) & 
                    (point_y <= target_box_y2) & 

                    # max distance lies within regression range
                    (max_dist >= layer_reg_range[0]) & 
                    (max_dist <= layer_reg_range[1]), 
                    True, False)
                
                foreground_mask = torch.any(mask, dim=0)        # H x W   (True iff point lies inside any box)
                background_mask = ~foreground_mask              # H x W   (True iff point doesn't lie in any box)
                
                # update positive samples
                N_foreground_points = foreground_mask.sum().detach()
                positive_samples += N_foreground_points

                # for each foreground point, find the bounding box area,
                # if point lies in multiple boxes, then multiple values 
                # across N_boxes will be positive. Select the one with
                # least area.
                area_per_point = mask * target_areas[:, None, None]             # N_boxes x H x W   (put box-area for foreground points)
                area_per_point[~mask] = BIG_NUMBER                              # N_boxes x H x W   (put BIG_NUMBER for background points)
                box_per_point = torch.min(area_per_point, dim=0)[1]             # H x W             (put box-index [0:N_boxes] of box with least area)
                box_per_point[background_mask] = -1                             # H x W             (put -1 for background and [0:N_boxes] for foreground points)

                label_per_point = target_label[box_per_point[foreground_mask]]  # 1 x N_foreground_points

                target_class_per_point = box_per_point.unsqueeze(dim=-1).repeat(1, 1, self.num_classes)             # H x W x N_classes
                target_class_per_point[background_mask] = 0                                                         # background = zeroes
                target_class_per_point[foreground_mask] = one_hot(label_per_point, num_classes=self.num_classes)    # foreground = one-hot
                target_class_per_point.detach()


                #####################
                # Classification Loss
                #####################

                # cls_logits_per_point      : (H x W x 20)
                # target_class_per_point    : (H x W x 20)
                cls_loss.append(sigmoid_focal_loss(cls_logits_per_point, target_class_per_point, reduction="sum"))

                if N_foreground_points == 0:
                    # skip regression and centerness loss if no foreground points found
                    continue

                #################
                # Regression Loss
                #################
                
                # reg_outputs_per_point     : (H x W x 4)
                predicted_l = reg_outputs_per_point[foreground_mask][:,0]       # (N_foreground,)     (predicted distance for each point in foreground)
                predicted_t = reg_outputs_per_point[foreground_mask][:,1]
                predicted_r = reg_outputs_per_point[foreground_mask][:,2]
                predicted_b = reg_outputs_per_point[foreground_mask][:,3]
                
                foreground_x = layer_points[foreground_mask][:,0]
                foreground_y = layer_points[foreground_mask][:,1]
                
                predicted_x1 = foreground_x - (predicted_l) * layer_stride
                predicted_y1 = foreground_y - (predicted_t) * layer_stride    
                predicted_x2 = foreground_x + (predicted_r) * layer_stride    
                predicted_y2 = foreground_y + (predicted_b) * layer_stride 
                predicted_xyxy = torch.stack((predicted_x1, predicted_y1, predicted_x2, predicted_y2), dim=1)       # (N_foreground, 4)
                
                target_xyxy = torch.zeros((*box_per_point.shape, 4), device=device)                        # (H x W x 4)
                target_xyxy[foreground_mask] = target_boxes[box_per_point[foreground_mask]]                         # (H x W x 4)
                target_xyxy = target_xyxy[foreground_mask]                                                          # (N_foreground, 4)
                target_xyxy.detach()

                # predicted_xyxy    : (N_foreground, 4)
                # target_xyxy       : (N_foreground, 4)
                reg_loss.append(giou_loss(predicted_xyxy, target_xyxy, reduction="sum"))


                #################
                # Centerness Loss
                #################

                # column vector that specifies the box-index of each foreground point
                box_per_foreground_point = box_per_point[foreground_mask].view(-1, 1)       # (N_foreground,)
                
                # permute l from (N_boxes x H x W) to (H x W x N_boxes) so that 
                # we can apply foreground mask of same starting dimensions (H x W)
                # Note that in (H x W x N_boxes), each pixel contains l-values for all boxes
                # Then we gather/select l-values of boxes corresponding to box-index specified in the index column vector
                l_foreground = l.permute(1,2,0)[foreground_mask]                            # (N_foreground, N_boxes)
                l_foreground = l_foreground.gather(1, box_per_foreground_point)             # (N_foreground,)

                # same for other distances
                t_foreground = t.permute(1,2,0)[foreground_mask].gather(1, box_per_foreground_point)
                r_foreground = r.permute(1,2,0)[foreground_mask].gather(1, box_per_foreground_point)
                b_foreground = b.permute(1,2,0)[foreground_mask].gather(1, box_per_foreground_point)
                
                target_centerness =  torch.sqrt(                                                            # (N_foreground,)
                    (torch.min(l_foreground, r_foreground) * torch.min(t_foreground, b_foreground)) / 
                    (torch.max(l_foreground, r_foreground) * torch.max(t_foreground, b_foreground))
                ).detach()

                # ctr_logits_per_point : (H x W x 1)
                predicted_centerness = ctr_logits_per_point[foreground_mask]                    # (N_foreground,1)
                ctr_loss.append(binary_cross_entropy_with_logits(predicted_centerness, target_centerness, reduction="sum"))


        # print(cls_loss, reg_loss, ctr_loss)
        
        positive_samples = max(1, positive_samples)

        # torch.stack(cls_loss) - N x H x W
        total_cls_loss = torch.sum(sum(cls_loss)) / positive_samples   # scalar
        total_reg_loss = torch.sum(sum(reg_loss)) / positive_samples   # scalar
        total_ctr_loss = torch.sum(sum(ctr_loss)) / positive_samples   # scalar
        final_loss = total_cls_loss + total_reg_loss + total_ctr_loss

        return {
            'cls_loss'  : total_cls_loss,
            'reg_loss'  : total_reg_loss,
            'ctr_loss'  : total_ctr_loss,
            'final_loss': final_loss
        }
            
        # positive_samples = 0
        # cls_loss, reg_loss, ctr_loss = torch.Tensor([0]).to("cuda"), torch.Tensor([0]).to("cuda"), torch.Tensor([0]).to("cuda")

        # nlayers = len(cls_logits)   # 3
        # for layer in range(nlayers):
        #     cls_logits[layer]   = cls_logits[layer].permute((0,2,3,1))      # N x H x W x 20
        #     reg_outputs[layer]  = reg_outputs[layer].permute((0,2,3,1))     # N x H x W x 4
        #     ctr_logits[layer]   = ctr_logits[layer].permute((0,2,3,1))      # N x H x W x 1

        # # targets       -> [ Dict {
        # #                       boxes: tensor(nboxes x 4), 
        # #                       labels: tensor(nboxes), 
        # #                       image_id: tensor(1), 
        # #                       area: tensor(1), 
        # #                       iscrowd: ??
        # #                   }] x nimages
        # # 
        # # points        -> [ meshgrid(x, y) ] x nlayers
        # # strides       -> tensor(nlayers)
        # # reg_range     -> tensor(nlayers x 2)
        # # cls_logits    -> [ tensor( N x H x W x 20) ] x nlayers
        # # reg_outputs   -> [ tensor( N x H x W x 4) ] x nlayers
        # # ctr_logits    -> [ tensor( N x H x W x 1) ] x nlayers

        # # for every layer
        # for layer, (layer_stride, layer_points, layer_reg_range, layer_cls_logits, layer_reg_outputs, layer_ctr_logits) in \
        #     enumerate(zip(strides, points, reg_range, cls_logits, reg_outputs, ctr_logits)):
        #     # layer_stride      -> tensor(1)
        #     # layer_points      -> tensor(H x W x 2)
        #     # layer_reg_range   -> tensor(1 x 2)
        #     # layer_cls_logits  -> tensor(N x H x W x 20)
        #     # layer_reg_outputs -> tensor(N x H x W x 4)
        #     # layer_ctr_logits  -> tensor(N x H x W x 1)

        #     # for every image in that layer
        #     for img, (img_cls_logits, img_reg_outputs, img_ctr_logits) in \
        #         enumerate(zip(layer_cls_logits, layer_reg_outputs, layer_ctr_logits)):
        #         # img_cls_logits    -> tensor(H x W x 20)
        #         # img_reg_outputs   -> tensor(H x W x 4)
        #         # img_ctr_logits    -> tensor(H x W x 1)
                
                
        #         # for every point in that image
        #         for point, point_cls_logit, point_reg_output, point_ctr_logit in \
        #             zip(layer_points.reshape(-1, 2), 
        #                 img_cls_logits.reshape(-1, 20), 
        #                 img_reg_outputs.reshape(-1, 4), 
        #                 img_ctr_logits.reshape(-1, 1)):

        #                 # point             -> tensor(2)
        #                 # point_cls_logit   -> tensor(20)
        #                 # point_reg_output  -> tensor(4)
        #                 # point_ctr_logit   -> tensor(1)

        #                 # find which target box contains this point
        #                 # choose the one with least area in case of clash
        #                 x, y = point
                        
        #                 target_box = None
        #                 target_center = None
        #                 target_label = None
        #                 target_area = float('inf')
        #                 target_box_ll = None
        #                 target_box_tt = None
        #                 target_box_rr = None
        #                 target_box_bb = None

        #                 # search which box contains the point(x,y)
        #                 for box, box_label, box_area in \
        #                     zip(targets[img]['boxes'], targets[img]['labels'], targets[img]['area']):

        #                     x1, y1, x2, y2 = box
        #                     center_x = (x1 + x2) / 2
        #                     center_y = (y1 + y2) / 2
        #                     center_box = torch.Tensor([center_x, center_y, center_x, center_y]).to('cuda') + (self.center_sampling_radius * layer_stride * torch.Tensor([-1,-1,1,1]).to('cuda'))
        #                     center_box.detach()

        #                     ll = (center_x - x1) / layer_stride
        #                     tt = (center_y - y1) / layer_stride
        #                     rr = (x2 - center_x) / layer_stride
        #                     bb = (y2 - center_y) / layer_stride
        #                     ll.detach()
        #                     tt.detach()
        #                     rr.detach()
        #                     bb.detach()
                            
        #                     # check if the reg_output lies within the reg_range
        #                     if layer_reg_range[0] <= max(ll, tt, rr, bb) <= layer_reg_range[1]:
                            
        #                         # check if point(x, y) lies within the target image's center box
        #                         if  center_box[0] <= x <= center_box[2] and \
        #                             center_box[1] <= y <= center_box[3] and \
        #                             box_area < target_area:
        #                             # save the box
        #                             target_box  = box
        #                             target_center = torch.Tensor([center_x, center_y, center_x, center_y]).to("cuda").detach()
        #                             positive_samples += 1
        #                             target_label = box_label.item() - 1     # convert to 0-indexed
        #                             target_area = box_area.item()
        #                             target_box_ll = ll
        #                             target_box_tt = tt
        #                             target_box_rr = rr
        #                             target_box_bb = bb
                                
        #                 # ###################
        #                 # classification loss
        #                 #   - points inside the sub-box are classified using one-hot encoding
        #                 #   - points outside the sub-box are considered to be background
        #                 # ###################
        #                 target_cls_logit = torch.zeros_like(point_cls_logit)    # tensor(20)
        #                 if target_label is not None:
        #                     target_cls_logit[target_label] = 1.0
                        
        #                 cls_loss += sigmoid_focal_loss(point_cls_logit, target_cls_logit, reduction="sum")
                        

        #                 # ###################
        #                 # regression loss & center-ness loss
        #                 #   - points inside the target-box are regressed
        #                 #   - points outside the target-box are ignored
        #                 # ###################
        #                 if target_label is not None:
        #                     x1, y1, x2, y2 = target_box
                            
        #                     reg_loss += giou_loss(target_center + (point_reg_output * layer_stride * torch.Tensor([-1,-1,1,1]).to('cuda')), target_box, reduction="sum")

        #                     # center-ness
        #                     # TODO: check this: sometimes one of pred l, r, t, b values are 0
        #                     pred_ctr =  torch.min(target_box_ll, target_box_rr) * torch.min(target_box_tt, target_box_bb)
        #                     pred_ctr /=  torch.max(target_box_ll, target_box_rr) * torch.max(target_box_tt, target_box_bb)
        #                     pred_ctr = torch.sqrt(pred_ctr).reshape(1).detach()
        #                     ctr_loss += binary_cross_entropy_with_logits(point_ctr_logit, pred_ctr)

        # positive_samples = max(positive_samples,1)
        # return {
        #     'cls_loss'  : cls_loss/positive_samples,
        #     'reg_loss'  : reg_loss/positive_samples,
        #     'ctr_loss'  : ctr_loss/positive_samples,
        #     'final_loss': (cls_loss + reg_loss + ctr_loss)/positive_samples
        # }

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) deocde the boxes
        (3) only keep boxes with scores larger than self.score_thresh
    (b) Combine all object candidates across levels and keep the top K (self.topk_candidates)
    (c) Remove boxes outside of the image boundaries (due to padding)
    (d) Run non-maximum suppression to remove any duplicated boxes
    (e) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from two different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels needed to be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        detections = []
        num_of_images = len(image_shapes)
        #TODO Can image be matricied
        for index in range(num_of_images):
            boxes = []
            scores = []
            labels = []
            combination = []
            for layer_stride, box_regression_per_image_per_level, logits_per_image_per_level, box_ctrness_image_per_level, points_per_level in zip(strides, reg_outputs, cls_logits, ctr_logits, points):
               
                box_regression_per_image = box_regression_per_image_per_level[index]
                logits_per_image = logits_per_image_per_level[index]
                box_ctrness_per_image = box_ctrness_image_per_level[index]
           
                scores_per_level = torch.sqrt(
                torch.sigmoid(logits_per_image) * torch.sigmoid(box_ctrness_per_image)).flatten()
                print(scores_per_level.shape)
               
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level
                topk_idxs = torch.where(keep_idxs)[0] ##INDEXES
               
                #TODO: Decode all boxes , then filtering ( reverse this )
                pred_l, pred_t, pred_r, pred_b = box_regression_per_image
                centerx = points_per_level[:,:,0]
                centery = points_per_level[:,:,1]
                x1 = centerx - pred_l * layer_stride 
                y1 = centery - pred_b * layer_stride
                x2 = centerx + pred_r * layer_stride 
                y2 = centery + pred_t * layer_stride
                boxes_per_level = clip_boxes_to_image(torch.stack([x1,y1,x2,y2]).permute(1,2,0) , image_shapes[index]).reshape(-1,4)
                box_idxs = torch.div(topk_idxs, 20, rounding_mode="floor")
               
                scores.append(scores_per_level[keep_idxs])
                boxes.append(boxes_per_level[box_idxs])
                labels.append(topk_idxs%20 + 1) #OFFSET append
             
            scores = torch.cat(scores, dim=0)
            boxes = torch.cat(boxes, dim=0)
            labels = torch.cat(labels, dim=0)  
            
            scores, idx = scores.topk(min(scores.size(0), self.topk_candidates))
            print(idx.shape, 'idx count')
            boxes = boxes[idx]
            labels = labels[idx]

            filtered_set = batched_nms(boxes, scores, labels, self.nms_thresh)
            
            filtered_set = filtered_set[: self.detections_per_img]
            print(labels[filtered_set].shape, "final length")
            detections.append(
                {
                    'boxes': boxes[filtered_set],
                    'scores': scores[filtered_set],
                    'labels': labels[filtered_set]
                }
            )
        return detections
