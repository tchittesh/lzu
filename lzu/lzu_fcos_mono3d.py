# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS

from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.single_stage_mono3d import (
    SingleStageMono3DDetector
)

from .invert_grid import invert_grid
from .fixed_grid import build_grid_generator


@DETECTORS.register_module()
class LZUFCOSMono3D(SingleStageMono3DDetector):
    """Fixed LZU + FCOSMono3D"""

    def __init__(self,
                 grid_generator,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(LZUFCOSMono3D, self).__init__(backbone, neck, bbox_head,
                                            train_cfg, test_cfg, pretrained)
        self.grid_generator = build_grid_generator(grid_generator)
        self.forward_grids = None  # cache forward warp "zoom" grids
        self.inverse_grids = None  # cache inverse warp "zoom" grids
        self.times = []

    def _get_scale_factor(self, ori_shape, new_shape):
        ori_height, ori_width, _ = ori_shape
        img_height, img_width, _ = new_shape
        w_scale = img_width / ori_width
        h_scale = img_height / ori_height
        assert w_scale == h_scale
        return w_scale

    def extract_feat(self, img, img_metas, **kwargs):
        """Directly extract features from the backbone+neck."""

        # "zoom" or forward warp input image
        if self.forward_grids is None:
            upsampled_grid, grid = self.grid_generator(
                img, img_metas, **kwargs)
            self.forward_grids = upsampled_grid[0:1], grid[0:1]
        else:
            upsampled_grid, grid = self.forward_grids
            B = img.shape[0]
            upsampled_grid = upsampled_grid.expand(B, -1, -1, -1)

        # Uncomment and change scale factor to run upsampling experiments
        # warped_imgs = F.interpolate(img, scale_factor=0.75)
        # warped_imgs = F.grid_sample(
        #     warped_imgs, upsampled_grid, align_corners=True)
        warped_imgs = F.grid_sample(img, upsampled_grid, align_corners=True)

        # Uncomment to visualize "zoomed" images
        # from mmcv import imdenormalize
        # from PIL import Image
        # import os
        # show_img = warped_imgs[0].permute(1, 2, 0).cpu().detach().numpy()
        # show_img = imdenormalize(
        #     show_img,
        #     mean=np.array([103.53, 116.28, 123.675]),
        #     std=np.array([1.0, 1.0, 1.0]),
        #     to_bgr=True)
        # img_name = os.path.basename(img_metas[0]['filename'])[:-4]
        # Image.fromarray(show_img.astype(np.uint8)).save(f'/project_data/ramanan/cthavama/FOVEAv2_exp/3D/lzu_fcos3d_sd/test_FT/vis_warped/{img_name}.png')  # noqa: E501
        # breakpoint()

        img_height, img_width = upsampled_grid.shape[1:3]
        img_shape = (img_height, img_width, 3)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = self._get_scale_factor(ori_shape, img_shape)

        # pad warped images; TODO: undo hardcoding size divisor of 32
        pad_h = int(np.ceil(img_shape[0] / 32)) * 32 - img_shape[0]
        pad_w = int(np.ceil(img_shape[1] / 32)) * 32 - img_shape[1]
        warped_imgs = F.pad(
            warped_imgs, (0, pad_w, 0, pad_h), mode='constant', value=0)
        pad_shape = (warped_imgs.shape[2], warped_imgs.shape[3], 3)

        # update img metas, assuming that all imgs have the same original shape
        for i in range(len(img_metas)):
            img_metas[i]['img_shape'] = img_shape
            img_metas[i]['scale_factor'] = scale_factor
            img_metas[i]['pad_shape'] = pad_shape
            img_metas[i]['pad_fixed_size'] = None
            img_metas[i]['pad_size_divisor'] = 32
            # resize ground truth boxes and centers
            if 'centers2d' in kwargs:
                kwargs['centers2d'][i] *= scale_factor
            if 'gt_bboxes' in kwargs:
                kwargs['gt_bboxes'][i] *= scale_factor
            for j in range(len(img_metas[i]['cam2img'][0])):
                img_metas[i]['cam2img'][0][j] *= scale_factor
                img_metas[i]['cam2img'][1][j] *= scale_factor

        # Encode
        warped_x = self.backbone(warped_imgs)
        if self.with_neck:
            warped_x = self.neck(warped_x)

        # Unzoom
        x = []
        # precompute and cache inverses
        separable = self.grid_generator.separable
        if self.inverse_grids is None:
            self.inverse_grids = []
            for i in range(len(warped_x)):
                input_shape = warped_x[i].shape
                inverse_grid = invert_grid(grid, input_shape,
                                           separable=separable)[0:1]
                self.inverse_grids.append(inverse_grid)
        # perform unzoom
        for i in range(len(warped_x)):
            B = len(warped_x[i])
            inverse_grid = self.inverse_grids[i].expand(B, -1, -1, -1)
            unwarped_x = F.grid_sample(
                warped_x[i], inverse_grid, mode='bilinear',
                align_corners=True, padding_mode='zeros'
            )
            x.append(unwarped_x)

        return tuple(x)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img, img_metas,
                              gt_bboxes=gt_bboxes, centers2d=centers2d)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              attr_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img, img_metas, **kwargs)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if self.bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
