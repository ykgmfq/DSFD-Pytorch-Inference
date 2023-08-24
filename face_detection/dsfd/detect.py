import typing

import numpy as np
import torch
from torch.hub import load_state_dict_from_url

from .. import torch_utils
from ..base import Detector
from ..build import DETECTOR_REGISTRY
from .config import resnet152_model_config
from .face_ssd import SSD

model_url = "https://folk.ntnu.no/haakohu/WIDERFace_DSFD_RES152.pth"


@DETECTOR_REGISTRY.register_module
class DSFDDetector(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dict = load_state_dict_from_url(
            model_url, map_location=self.device, progress=True
        )
        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _detect(
        self,
        x: torch.Tensor,
    ) -> typing.List[np.ndarray]:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            boxes = self.net(x, self.confidence_threshold, self.nms_iou_threshold)
        return boxes
