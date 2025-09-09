import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# For modeling I will use a more complex approach using CNNs instead of normal networks

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),                           
    Dense(128, activation='relu'),                          
    Dense(10, activation='softmax')                         
])

# Compiling the model

model.compile(optimizer=Adam(learning_rate=0.001),       # Used tensorflow.keras.optimizers                  
              loss='sparse_categorical_crossentropy',      
              metrics=['accuracy'])                        

# Training



model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

model.save('digit_model.h5')                                   

#Evaluate

loss, accuracy = model.evaluate(x_test, y_test)             

print(f'Test accuracy: {accuracy * 100:.2f}%')







