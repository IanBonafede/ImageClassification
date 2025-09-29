import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load in the data
data = keras.datasets.fashion_mnist

#seperate data into taining and testing
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#these labels are 0-9
#we must define what they mean

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#data is saved as pixel values, we can shrink data by dividing by max value, new range 0-1
train_images = train_images/255.0
test_images = test_images/255.0

#we are using 28x28 images, flattened is 784 long input layer
#one 128 long hidden layer
#10 long (0-9) output layer

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax") #makes the total value of each prediction = 1 so 0.05 + 0.1 + 0.85
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#training the model
#epochs will say how many times it will go through the same data in different orders
#because the order they data points are seen affects the direction of the gradient
model.fit(train_images, train_labels, epochs=5)

#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested Acc: ", test_acc)'


#predictions
#get all predictions for each class
prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()