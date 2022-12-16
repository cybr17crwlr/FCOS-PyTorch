import torch
from torch import nn


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PointGenerator(nn.Module):
    """
    A generator for 2D "points".

    Args:
        img_max_size (int): the longest side of an input image
        fpn_stride List[int]: feature stride on each pyramid level
        regression_range List[float]: regression range on each pyramid level
    """

    def __init__(
        self,
        img_max_size,
        fpn_strides,  # strides of fpn levels
        regression_range,  # regression range (on feature grids)
    ):
        super().__init__()
        # sanity check, # fpn levels and length divisible
        fpn_levels = len(fpn_strides)
        assert len(regression_range) == fpn_levels

        # save params
        self.max_size = img_max_size
        self.fpn_levels = fpn_levels
        self.register_buffer(
            "fpn_strides", torch.as_tensor(fpn_strides, dtype=torch.float)
        )
        self.register_buffer(
            "regression_range", torch.as_tensor(regression_range, dtype=torch.float)
        )

        # generate all points and buffer the list
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # loop over all points at each pyramid level
        for l, stride in enumerate(self.fpn_strides):
            points = torch.arange(0, self.max_size, stride)
            points += 0.5 * stride
            grid_x, grid_y = torch.meshgrid(points, points, indexing="ij")
            grids = torch.stack([grid_x, grid_y], dim=-1)
            # size: H x W x 2 (height, width)
            points_list.append(grids)

        return BufferList(points_list)

    """
    Calling the forward function will return
    (1) a list of tensors with each of size H x W x 2, which records the
    2D coordinates of the feature map.
    (2) the feature stride on each pyramid level
    (3) the regression range on each pyramid level
    """
    def forward(self, feats):
        # feats will be a list of torch tensors
        torch._assert(len(feats) == self.fpn_levels, "FPN levels mismatch")
        pts_list = []
        for feat, buffer_pts in zip(feats, self.buffer_points):
            h, w = feat.shape[-2], feat.shape[-1]
            torch._assert(
                (h <= buffer_pts.shape[0]) and (w <= buffer_pts.shape[1]),
                "Input feature size larger than max size",
            )
            pts = buffer_pts[:h, :w, :].clone()
            pts_list.append(pts)
        return pts_list, self.fpn_strides, self.regression_range
