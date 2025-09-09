import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore

# Loading the digits as NumPy arrays and normalizing
# for the 0-255 grayscale images in a [0, 1] range

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Building the model

model = Sequential([
    Flatten(input_shape=(28,28)),                           # Choosed a 28x28 (784 neurons) because images are 28x28 pixels
    Dense(128, activation='relu'),                          # Fully connected layer with 128 neurons and relu (rectified linear unit) for gradient aproximation
    Dense(10, activation='softmax')                         # Output layer with 10 neurons (digits from 0-9) and softmax that converts scores to probabilities
])

# Compiling the model

model.compile(optimizer='adam',                             # Adaptive gradient method
              loss='sparse_categorical_crossentropy',       # Expects integer labels (0-9)
              metrics=['accuracy'])                         # Report classification accuracy during training/evaluation

# Training

# Trains for 5 passes (epochs) over the training set
# Uses (x_test, y_test) as validation data to monitor performance each epoch
# This function returns a History object containing loss/accuracy curves
# Default batch_size is 32; data is shuffled each epoch

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save('digit_model.h5')                                   # Save the trained model to disk in HDF5 format

#Evaluate

loss, accuracy = model.evaluate(x_test, y_test)             # Runs the model on the test set to compute final loss and accuracy

print(f'Test accuracy: {accuracy * 100:.2f}%')







