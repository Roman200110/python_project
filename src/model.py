import torch
import torchvision.models as models
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torch import nn


class AnimalClassifier:
    """
    A class to represent the FasterRCNN Model.

    Attributes
    ----------
    num_classes : int
        number of classes in the dataset

    Methods
    -------
    get_model():
        Returns the FasterRCNN model.
    """

    def __init__(self, num_classes):
        """
        Constructs all the necessary attributes for the FasterRCNN object.

        Parameters
        ----------
            num_classes : int
                number of classes in the dataset
        """
        # Use the recommended way to load pretrained weights
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def get_model(self):
        """Returns the FasterRCNN model."""
        return self.model
