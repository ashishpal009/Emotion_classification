# Emotion Classification using Deep Learning

## Overview
This project involves building a Convolutional Neural Network (CNN) model using TensorFlow/Keras to classify facial expressions into 7 categories. The dataset comprises grayscale images resized to 48x48 pixels for training and testing.

## Project Structure
1. **Data Preparation**:
   - Images are loaded using `ImageDataGenerator` with data augmentation for training and validation.
   - Training and validation datasets are organized into separate directories.

2. **Model Architecture**:
   - A sequential CNN model is built with multiple convolutional, pooling, and dropout layers.
   - Batch normalization is applied for better training stability.
   - A final dense layer with softmax activation outputs probabilities for 7 emotion categories.

3. **Training**:
   - The model is compiled using Adam optimizer and categorical crossentropy loss.
   - Training involves 100 epochs with validation metrics tracked.

4. **Evaluation**:
   - Training and validation metrics are plotted for accuracy and loss.
   - Generalization metrics (accuracy and loss) are computed for the training and validation datasets.

5. **Prediction**:
   - A test image is manually loaded, preprocessed, and passed through the trained model for emotion prediction.

## Files and Directories
- **`train_dir`**: Directory containing the training dataset.
- **`test_dir`**: Directory containing the testing dataset.
- **`model_fer2013.h5`**: Saved model file.

## Prerequisites
- Python 3.x
- TensorFlow/Keras
- Matplotlib, Seaborn
- Google Colab (for execution with GPU support)

## How to Run
1. Mount Google Drive to access the dataset.
2. Ensure the dataset directory structure is as follows:
   ```plaintext
   /train
       /angry
       /disgust
       ...
   /test
       /angry
       /disgust
       ...
   ```
3. Run the notebook step by step in Google Colab.

## Key Results
- Model achieves accuracy on training and validation datasets.
- Visualizations highlight the training process.

## Limitations
- Works with grayscale images of size 48x48 pixels.
- Requires further tuning for real-world applications.
