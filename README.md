# README.txt

This directory contains the raw dataset for training, validating, and testing a machine learning model that predicts the type of tank (light, middle, heavy, anti-tank, artillery) based on images.

## Directory Structure:

The dataset is organized into three main folders: `train`, `val`, and `test`. Each of these folders contains subdirectories for each tank class. The subdirectories are named according to the type of tank and contain the respective images.

## Instructions:

1. **Raw Data Preparation**:

   - Place all raw images into the appropriate subdirectories within the `raw` folder based on their class.

2. **Dataset Splitting**:

   - Run the provided Python script `split_data.py` to automatically split the raw data into training, validation, and test sets. The script will distribute the images according to the specified ratios and copy them into the `train`, `val`, and `test` folders.

3. **Training the Model**:

   - Use the training images from the `train` folder to train your machine learning model.
   - Validate the model using the images from the `val` folder during training to monitor performance and prevent overfitting.

4. **Testing the Model**:
   - After training, evaluate the final model performance using the images from the `test` folder.
