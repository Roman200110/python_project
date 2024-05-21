import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes
from data_loader import get_transforms
from model import AnimalClassifier
from utils import load_checkpoint
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os

def load_image(image_path):
    """
    Load an image from the given path and convert it to a tensor.

    Parameters
    ----------
    image_path : str
        Path to the image file

    Returns
    -------
    image : torch.Tensor
        The image loaded as a tensor
    """
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    return image


def get_detections(model, image_path, device, classes, threshold=0.8):
    """
    Get detections for an image using the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model
    image_path : str
        Path to the image file
    device : torch.device
        The device to run the model on (CPU or GPU)
    classes : list
        List of class names
    threshold : float
        Confidence threshold for displaying bounding boxes

    Returns
    -------
    detections : dict
        A dictionary containing bounding boxes and corresponding labels
    """
    model.eval()
    image = load_image(image_path).to(device)
    with torch.no_grad():
        prediction = model([image])
        pred = prediction[0]

    boxes = pred['boxes'][pred['scores'] > threshold]
    labels = [classes[i] for i in pred['labels'][pred['scores'] > threshold].tolist()]

    return {'boxes': boxes, 'labels': labels}


def demo_image(image_path, model, device, classes, threshold=0.8):
    """
    Run the model on an image and display the result with bounding boxes.

    Parameters
    ----------
    image_path : str
        Path to the image file
    model : torch.nn.Module
        The trained model
    device : torch.device
        The device to run the model on (CPU or GPU)
    classes : list
        List of class names
    threshold : float
        Confidence threshold for displaying bounding boxes
    """
    detections = get_detections(model, image_path, device, classes, threshold)
    image = load_image(image_path)

    img_int = torch.tensor(image * 255, dtype=torch.uint8)
    fig = plt.figure(figsize=(14, 10))
    plt.imshow(draw_bounding_boxes(
        img_int, detections['boxes'],
        detections['labels'], width=4
    ).permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    dataset_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/data/raw/TAWIRI 02.v3i.coco"  # Update this to your dataset path
    num_classes = len(COCO(os.path.join(dataset_path, "test", "_annotations_bbox.coco.json")).cats.keys())
    classes = [v['name'] for k, v in COCO(os.path.join(dataset_path, "test", "_annotations_bbox.coco.json")).cats.items()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AnimalClassifier(num_classes).get_model().to(device)

    # Load the trained checkpoint
    checkpoint_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/checkpoints/model_epoch_9.pth"  # Update this to the actual checkpoint path
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9,
                                weight_decay=1e-4)
    load_checkpoint(checkpoint_path, model, optimizer)

    # Specify the image path for testing
    image_path = dataset_path + "/test/3-elephants-from-above_jpg.rf.48b8cb1d7512b394b0132db0db3f39d4.jpg"  # Update this to your test image path
    demo_image(image_path, model, device, classes)
