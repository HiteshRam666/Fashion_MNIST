# 🧥 Fashion MNIST Classification with CNN

This repository contains a Jupyter Notebook that demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset consists of 70,000 grayscale images of 28x28 pixels, divided into 10 different fashion categories.

## 📂 Files

- **Fashion_MNIST.ipynb**: The main Jupyter Notebook containing the code for loading the dataset, building the CNN model, training the model, and evaluating its performance.

## 🚀 Getting Started

### Prerequisites

To run this notebook, you need to have Python and Jupyter Notebook installed, along with the following Python libraries:

- TensorFlow
- Keras
- Matplotlib
- NumPy

You can install these libraries using pip:

```bash
pip install tensorflow keras matplotlib numpy
```

### Running the Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/fashion-mnist-cnn.git
   ```

2. Navigate to the directory:

   ```bash
   cd fashion-mnist-cnn
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `Fashion_MNIST.ipynb` and run the cells to train and evaluate the model.

## 📊 Model Architecture

The CNN model used in this notebook consists of the following layers:

1. **Input Layer**: Reshapes the 28x28 grayscale images to a format suitable for the model.
2. **Convolutional Layers**: Two sets of convolutional layers with ReLU activation and MaxPooling layers to extract features from the images.
3. **Flatten Layer**: Flattens the output from the convolutional layers to feed into the dense layers.
4. **Fully Connected Layers (Dense Layers)**: Two dense layers to learn from the extracted features.
5. **Output Layer**: A dense layer with 10 units (one for each class) and softmax activation to predict the class probabilities.

## 📈 Model Training

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

The model is trained on the Fashion MNIST training set and validated on the test set. 

## 🔍 Evaluation

The notebook includes code to evaluate the model's performance using the test dataset. It displays the model's accuracy and loss, and provides a confusion matrix to visualize the classification performance across different fashion categories.

## 🎨 Visualizations

The notebook provides several visualizations to help understand the model's performance, including:

- Sample images from the Fashion MNIST dataset.
- Accuracy and loss plots for training and validation data over epochs.
- Confusion matrix to show model performance across all classes.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
