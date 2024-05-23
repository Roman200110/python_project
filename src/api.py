import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from torchvision import transforms as T
from model import AnimalClassifier
from utils import load_checkpoint
import os
import io
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

app = FastAPI()

# Setup the dataset and model parameters
dataset_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/data/raw/TAWIRI 02.v3i.coco"  # Update this to your dataset path
num_classes = len(COCO(os.path.join(dataset_path, "test", "_annotations_bbox.coco.json")).cats.keys())
classes = [v['name'] for k, v in COCO(os.path.join(dataset_path, "test", "_annotations_bbox.coco.json")).cats.items()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnimalClassifier(num_classes).get_model().to(device)

# Load the trained checkpoint
checkpoint_path = "/home/roman/roman/master_lessons/second_semester/python 2/python_project/checkpoints/model_epoch_20.pth"  # Update this to the actual checkpoint path
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9,
                            weight_decay=1e-4)
epoch = load_checkpoint(checkpoint_path, model, optimizer)
print(f'Loaded checkpoint from epoch {epoch}')


def load_image(image_file):
    """
    Load an image from the given file and convert it to a tensor.

    Parameters
    ----------
    image_file : file-like object
        The image file to be loaded

    Returns
    -------
    torch.Tensor
        The loaded image as a tensor
    """
    image = Image.open(image_file).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image = transform(image)
    return image


def get_detections(model, image, device, classes, threshold=0.8):
    """
    Get detections for an image using the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model
    image : torch.Tensor
        The image tensor
    device : torch.device
        The device to run the model on (CPU or GPU)
    classes : list
        List of class names
    threshold : float
        Confidence threshold for displaying bounding boxes

    Returns
    -------
    dict
        A dictionary containing bounding boxes and corresponding labels
    """
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        prediction = model([image])
        pred = prediction[0]

    boxes = pred['boxes'][pred['scores'] > threshold]
    labels = [classes[i] for i in pred['labels'][pred['scores'] > threshold].tolist()]

    return {'boxes': boxes, 'labels': labels}


@app.get("/")
def read_root():
    """
    Root endpoint for health check.

    Returns
    -------
    JSONResponse
        A response indicating that the API is running
    """
    return JSONResponse(content={"message": "API is running"}, status_code=200)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload an image file and perform object detection.

    Parameters
    ----------
    file : UploadFile
        The uploaded image file

    Returns
    -------
    StreamingResponse
        A response containing the processed image with bounding boxes
    """
    try:
        image = load_image(file.file)
        detections = get_detections(model, image, device, classes)

        img_int = torch.tensor(image * 255, dtype=torch.uint8)
        img_with_boxes = draw_bounding_boxes(img_int, detections['boxes'], detections['labels'], width=4).permute(1, 2,
                                                                                                                  0).cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img_with_boxes)
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
