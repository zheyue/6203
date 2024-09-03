
# Age, Gender, and Ethnicity Prediction and Classification - CNN

**Group 1: Rayna Liu, Zheyue Wang**

## Introduction

In recent years, facial image analysis has gained significant importance across various applications, such as biometrics, security systems, and entertainment. This project leverages Convolutional Neural Networks (CNNs) for the tasks of age estimation, ethnicity classification, and gender classification. CNNs have demonstrated impressive performance in face recognition, image classification, and object detection tasks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [How It Works](#how-it-works)
4. [Model Selection and Training](#model-selection-and-training)
5. [Running the Code](#running-the-code)
6. [Results and Analysis](#results-and-analysis)
7. [Additional Resources](#additional-resources)

## Project Overview

This project aims to build CNN models to classify age, gender, and ethnicity from facial images. We utilized two separate CNN models: one for age estimation and another for ethnicity and gender classification. The models were trained on a dataset of facial images and evaluated for their accuracy and performance in predicting the desired attributes.

## File Structure

The repository contains the following folders and files:

- **Group-Proposal**
  - `Group Proposal.pdf`: Describes our plan for the final project.

- **Final-Group-Project-Report**
  - `Final Group Project Report`: The final report detailing the project methodology, results, and conclusions.

- **Final-Group-Project-Appendix**

- **Final-Group-Presentation**
  - `Final Group Presentation.pdf`: The PowerPoint presentation summarizing our group’s work.

- **Code**
  - `Final_Project_Age_Ethnicity_Gender_Model.py`: The main script for training the models, making predictions, and analyzing results.
  - `Reload_Model_for_Analysis.py`: A script for loading the best models and performing additional analysis.

- **zheyue-wang-individual-project**
  - `Code`: Individual code contributions.
  - `zheyue-wang-final-project-report`: Individual report detailing personal contributions and findings.

## How It Works

1. **Read the README.md in the Code folder**: This provides additional context and prerequisites for running the code.
2. **Run `Final_Project_Age_Ethnicity_Gender_Model.py`**: This script trains the CNN models on the dataset and evaluates their performance.
3. **Run `Reload_Model_for_Analysis.py`**: This script reloads the trained models and performs further analysis.
4. **Compare the results**: Review the results against those presented in the final report.
5. **Review the presentation**: Look at `Final Group Presentation.pdf` for a summary of the project's key points.

## Model Selection and Training

### Age Estimation Model

For the age estimation model, we used Keras to construct a CNN model with the following details:

- **Input Shape**: (48, 48, 1) due to the 48x48 grayscale images.
- **Architecture**: The model consists of 11 convolutional layers.
  - The first 4 layers utilize max pooling and batch normalization to reduce both time and space complexity and handle vanishing/exploding gradients.
  - The subsequent 7 convolutional layers have 512 neurons, a 3x3 kernel size, 1x1 stride size, and zero padding.
  - ReLU is used as the activation function due to the regression nature of the age estimation task, as it only returns positive values.
- **Dropout**: A dropout rate of 0.5 was applied to prevent overfitting.
- **Flatten and Fully Connected Layers**: The model includes 4 fully connected layers, including the target layer for final predictions.

### Ethnicity and Gender Classification Model

For the ethnicity and gender classification model, we also used Keras to construct a CNN model with the following details:

- **Input Shape**: (48, 48, 1) due to the 48x48 grayscale images.
- **Architecture**: The model consists of 6 convolutional layers.
  - The first 2 layers utilize pooling and batch normalization to reduce complexity and handle vanishing/exploding gradients.
  - The subsequent 4 layers also incorporate batch normalization.
  - ReLU is used as the activation function.
  - A sigmoid function is applied in the output layer to convert the model’s output into probability scores, facilitating interpretation for multi-label classification.
- **Dropout**: We used a dropout layer to prevent overfitting, with 4096 neurons in one of the dense layers.
- **One-Hot Encoding**: Ethnicity and gender labels were concatenated into a one-hot encoded 7-label format. For example, if an image represents a white female, the target would be [1,0,0,0,0,0,1].

## Running the Code

1. Ensure all dependencies are installed. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the dataset as described in the `README.md` file in the Code folder.

3. Execute the scripts in the following order:
   - `python Final_Project_Age_Ethnicity_Gender_Model.py`
   - `python Reload_Model_for_Analysis.py`

4. Review the output and compare with the results in the final report.

## Results and Analysis

Refer to the final report for detailed results and analysis. Key findings include [briefly summarize key findings].

## Additional Resources

- **Dataset Information**: The dataset used in this project was downloaded from Kaggle. Please refer to the project documentation for more details on the dataset preparation and preprocessing steps.
- **Model Architecture Details**: Detailed descriptions of the model architecture can be found in the `Final Project.pdf` document included in this repository.
- **Preprocessing and Augmentation Techniques**: See the `Final Project.pdf` for information on the data preprocessing and augmentation techniques employed in this project.

---
