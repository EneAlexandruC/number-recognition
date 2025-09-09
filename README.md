# Handwritten Digit Classifier App

This is a **machine learning prototype app** for practicing **image classification**. The app solves the classical problem of predicting digits (0-9) from images of handwritten numbers.

---

## Dataset

I used the **MNIST dataset**, which is included in TensorFlow. The dataset contains:

- **60,000 labeled images** for training  
- **10,000 images** for testing  
- Each image is a **28x28 grayscale digit**

---

## Installation

For an easy setup, I installed **Anaconda** because it comes with most ML libraries pre-installed.

---

## Frameworks & Tools

- **Python 3.13.3**  
- **TensorFlow + Keras** (for building neural networks)  
- **Flask** (for backend)  
- **VS Code** (IDE)

---

## Model Development

1. **Basic Neural Network (Dense Layers)**  
   - Achieved **96% accuracy**

2. **Convolutional Neural Network (CNN)**  
   - Optimized model achieved **>99% accuracy**

---

## Future Plans

- Implement the model as a **mobile app**  
- Use the **camera** to take pictures of handwritten digits and predict them in real-time  
- This project currently serves as a **prototype**

---

## How to Run

```bash
# Install dependencies
pip install tensorflow flask numpy

# Run the Flask app
python app.py