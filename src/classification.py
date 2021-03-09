import os
from pathlib import Path
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np


class Classifier:

    def __init__(self):
        """
        Creates a new Classifier object. A classifier can be used to either train a new machine learning model or use an
        already existing model to predict values for new image data.
        """
        self.model = None

    def train(self):
        """
        Uses the MNIST data set to train CNN model. After training the model is saved as 'model.h5' and can be used for
        predictions.
        """
        # Load MNIST data set, normalize values and split into train and test subsets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
        X_train = X_train / 255
        X_test = X_test / 255

        # Set up output as an 1-hot-array
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]

        # Define the model
        def create_model():
            model = Sequential()
            model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Conv2D(15, (3, 3), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))
            # Compile model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        # Train the model
        self.model = create_model()
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
        # Evaluate the model
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Error: %.2f%%" % (100 - scores[1] * 100))

        self.model.save("model.h5")

    def predict(self, image):
        """
        Uses a classification model to predict what number is depicted on an image. The image has to
        contain a white number on a black background to conform with the visuals the model was trained on.

        :param image: an 28x28 pixel sized image which will be predicted
        :return: a tuple containing the predicted number and the certainty of it
        """
        # Check if a trained model exists
        if not self.model:
            try:
                self.model = load_model("src/model.h5")
            except IOError:
                print("No model found. Trying to train a new one.")
                self.train()

        # Converting the image into a numerical representation
        rgba_array = np.asarray(image)  # 28x28x4 int nd array holding rgba values
        saturation_array = []
        for rgba_row in rgba_array:
            opacity_row = []
            for rgba_cell in rgba_row:
                opacity_row.append(rgba_cell[0])  # Rgb values are all equal because the image is in black and white
            saturation_array.append(opacity_row)

        saturation_array = np.array(saturation_array)  # 28x28 int nd array holding opacity values -> black/white value
        float_array = np.array([saturation_array])  # Convert into float nd array
        float_array = float_array.reshape((float_array.shape[0], 28, 28, 1)).astype('float32')

        normalized_array = float_array / 255  # Normalize values: 0 - 255 -> 0 - 1
        prediction = self.model.predict(normalized_array)[0]
        predicted_number = np.argmax(prediction)
        certainty = float(np.max(prediction))
        return predicted_number, certainty


if __name__ == "__main__":
    c = Classifier()
    c.train()
