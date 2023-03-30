from mmdet.models.builder import HEADS
from mmdet3d.models.dense_heads import FCOSMono3DHead


@HEADS.register_module()
class FCOSMono3DHeadNoRescale(FCOSMono3DHead):
    """Same as original head, except ignores rescaling at end."""

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           dir_cls_preds,
                           attr_preds,
                           centernesses,
                           mlvl_points,
                           input_meta,
                           cfg,
                           rescale=False):
        return super()._get_bboxes_single(cls_scores, bbox_preds,
                                          dir_cls_preds, attr_preds,
                                          centernesses, mlvl_points,
                                          input_meta, cfg, rescale=False)
