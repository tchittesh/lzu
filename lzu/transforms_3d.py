from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize


@PIPELINES.register_module()
class Resize3D(Resize):

    def __call__(self, input_dict):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map, *and additionally adjust the camera intrinsics*.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        input_dict = super().__call__(input_dict)
        w_scale, h_scale, _, _ = input_dict['scale_factor']
        assert w_scale == h_scale

        input_dict['scale_factor'] = w_scale
        if 'centers2d' in input_dict:
            input_dict['centers2d'] *= w_scale
        for i in range(len(input_dict['cam2img'][0])):
            input_dict['cam2img'][0][i] *= w_scale
            input_dict['cam2img'][1][i] *= h_scale
        return input_dict
