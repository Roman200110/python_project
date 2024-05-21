# Animal Classification Project

## Overview
This project is an animal classification system using a Convolutional Neural Network (CNN) implemented in PyTorch. The model classifies images into different animal categories. The project follows object-oriented programming principles, includes detailed docstrings, and is designed with modularity and reusability in mind. Additionally, it includes a web API built with FastAPI for serving predictions.

## Project Structure
```
animal-classification/
├── data/
│   ├── raw/
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data_loader.py
│   ├── demo.py
│   ├── model.py
│   ├── trainer.py
│   ├── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_model.py
│   ├── test_trainer.py
│   ├── test_utils.py
├── train.py
├── README.md
├── requirements.txt
```

## Setup
### Prerequisites
- Python 3.8+
- pip

### Install Required Packages
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/animal-classification.git
    cd animal-classification
    ```

2. Set up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Download Dataset
1. Ensure you have downloaded the dataset from [roboflow](https://universe.roboflow.com/wild-animals-datasets/tawiri-02/dataset/3) and placed it in the `data/raw` directory.

### Training the Model
Run the training script:
```bash
python train.py
```

### Testing
Run the tests using Pytest:
```bash
pytest
```

### API
Start the API server:
```bash
uvicorn src.api:app --reload
```

Use the `/predict` endpoint to classify images. You can test this using an API client like Postman or curl.

Example using curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path_to_your_image.jpg"
```

### Code Quality
This project adheres to the Google Python Style Guide. Docstrings are included to describe classes, methods, and functions. The code is modular and designed for reusability.

### Code Structure
- `data_loader.py`: Handles data loading and preprocessing.
- `model.py`: Used Faster R-CNN for object detection.
- `trainer.py`: Contains the training and evaluation logic.
- `train.py`: Script to run the training and evaluation.

### Tests
- `test_data_loader.py`: Tests for data loading functionality.
- `test_model.py`: Tests for model architecture and forward pass.
- `test_trainer.py`: Tests for training and evaluation routines.

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

## Contributing
If you would like to contribute, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
