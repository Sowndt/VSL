# Vietnamese Sign Language Recognition Project

## Project Description
This project focuses on building a Sign Language Recognition (SLR) system using machine learning and deep learning tools. The goal is to extract features from images or videos containing hand gestures and body poses, then train a model to classify these gestures into corresponding signs or words. The project utilizes the Mediapipe library for landmark detection and applies data augmentation techniques to enhance model performance. Instead of predicting an entire sequence of frames, this project concentrates on recognizing static images and combining them to form multi-motion signs, reducing computation time and ensuring compatibility with moderate and low-spec configurations.

## Key Features
- Extract hand landmarks (21 points) and pose landmarks (33 points) from images using Mediapipe.
- Apply data augmentation (translation, affine transformation, rotation) to enhance the dataset.
- Store data in a `data.pickle` file with separate features and labels.
- Train deep learning models for gesture recognition.
- Support basic word segmentation logic from landmark sequences.

## Directory Structure
```
VSL/
├── collect_imgs.py # Script to capture images for dataset creation
├── keypoint_extract.py # Module for landmark extraction
├── create_dataset.py # Script to process images and create data
├── label_binarizer.pkl # Pickle file containing dataset labels
├── train_model.ipynb # Script to train the model
├── Final_model.h5 # Pre-trained model file
├── TEST.py # Script for real-time model testing
└── README.md         # Project description file
```

## Installation Requirements
- Python 3.10+
- Required libraries:
  ```bash
  pip install mediapipe opencv-python numpy pickle tensorflow scikit-learn
  ```

## Usage Instructions

### Using the Pre-trained Model:
**Clone this GitHub repository and run `TEST.py`**
  ```bash
     git clone https://github.com/Sowndt/VSL
  ```
#### DEMO

![Image](https://github.com/user-attachments/assets/d350a57a-2dea-46e2-968f-9b42855a2b10)

![Image](https://github.com/user-attachments/assets/c4dad0f2-2707-43d8-abe6-3142589bf48c)

### Create Your Own Dataset
1. **Prepare Data**:
   - Create a directory and use `collect_imgs.py` to capture and save images into the newly created folder. Assign labels to each subfolder (e.g., `a`, `b`, `cam_on`), note: Vietnamese labels are not allowed as they will cause errors.
   - Run `create_dataset.py` to preprocess data and create `data.pickle`:
     ```bash
     python create_dataset.py
     ```

2. **Train the Model**:
   - Use `train_model.ipynb` to train the model:
     ```bash
     python train_model.ipynb
     ```

3. **Test Real-time Recognition**:
   - Run `TEST.py` to activate the camera and perform real-time sign language recognition for trained gestures:
     ```bash
     python Test.py
     ```

## Notes
- Ensure all image files are in `.jpg`, `.jpeg`, or `.png` format and are not corrupted.
- Adjust `PREDICTION_INTERVAL` in `TEST.py` to change the frame interval for model predictions (set lower for high-spec machines and higher for low-spec machines to ensure a smooth experience).
- Adjust the recognition time for each sign in `TEST.py` to match the signing speed of individuals:
     ```Python
     word_buffer[:] = [(w, t) for w, t in word_buffer if current_time - t <= 2]
     ```

## Updates
- Update Date: 23/07/2025
- Current Version: 1.0
