from settings import *
from random import shuffle
from cv2 import imread, IMREAD_GRAYSCALE, bitwise_not
from os import path, listdir

import pandas as pd
import numpy as np
import pickle


def create_training_data_from_images(class_names, data_path):
    training_data = []

    # For each class name in the list of class names
    for class_name in class_names:
        # Build path to images for the class and the corresponding class number (label)
        img_path = path.join(data_path, class_name)
        class_num = class_names.index(class_name)

        # For each image in the class directory
        for img in listdir(img_path):
            try:
                # Load image as a greyscale image array
                img_array = imread(path.join(img_path, img), IMREAD_GRAYSCALE)

                # Convert from white to black background and normalise
                img_array = bitwise_not(img_array)
                img_array = img_array / 255.0

                # Reshape image to correct size and put data together
                img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                training_data.append([img_array, class_num])
            except Exception as e:
                print(e)
                pass

    # Shuffle data
    shuffle(training_data)

    x_out = []
    y_out = []

    # Split data into images and labels
    for features, label in training_data:
        x_out.append(features)
        y_out.append(label)

    # Convert to numpy arrays and final reshape
    x_out = np.array(x_out).reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    y_out = np.array(y_out)

    return x_out, y_out


def create_training_data_from_csv(file_path, offset):
    # Load Data
    train = pd.read_csv(file_path)

    # Separating Data and Label
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)

    # Normalize the data
    X_train = X_train / 255.0

    # Reshape the array into 28 x 28 pixel
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Putting data together after reshaping
    training_data = []
    for i in range(0, len(X_train)):
        training_data.append([X_train[i], Y_train[i]])

    # Shuffle data
    shuffle(training_data)

    x_out = []
    y_out = []

    # Separate data back into labels and data
    for features, label in training_data:
        x_out.append(features)
        y_out.append(label - offset)  # offset of 10 for chars, 0 for modifiers, 60 for compound

    # Final reshape of images
    x_out = np.array(x_out).reshape((-1, IMG_SIZE, IMG_SIZE, 1))

    return x_out, y_out


def save_numpy_data(in_x, in_y, file_path):
    in_x = np.array(in_x)
    in_y = np.array(in_y)

    pickle_out = open(file_path + "x.pickle", "wb")
    pickle.dump(in_x, pickle_out)
    pickle_out.close()

    pickle_out = open(file_path + "y.pickle", "wb")
    pickle.dump(in_y, pickle_out)
    pickle_out.close()


def initialise_ekush():
    # Create Ekush male character dataset pickle using offset of 10
    x_one, y_one = create_training_data_from_csv(EKUSH_DIR + "male/chars/csv/malechar1.csv", 10)
    x_two, y_two = create_training_data_from_csv(EKUSH_DIR + "male/chars/csv/malechar2.csv", 10)
    x = np.concatenate((x_one, x_two))
    y = np.concatenate((y_one, y_two))
    save_numpy_data(x, y, EKUSH_DIR + "male/chars/")

    # Create Ekush male modifier dataset pickle using offset of 0
    x, y = create_training_data_from_csv(EKUSH_DIR + "male/modifiers/csv/maleModifiers.csv", 0)
    save_numpy_data(x, y, EKUSH_DIR + "male/modifiers/")

    # Create Ekush male compound character dataset pickle using offset of 60
    x_one, y_one = create_training_data_from_csv(EKUSH_DIR + "male/compound/csv/maleCompunds1.csv", 60)
    x_two, y_two = create_training_data_from_csv(EKUSH_DIR + "male/compound/csv/maleCompunds2.csv", 60)
    x = np.concatenate((x_one, x_two))
    y = np.concatenate((y_one, y_two))
    save_numpy_data(x, y, EKUSH_DIR + "male/compound/")

    # Create Ekush female character dataset pickle using offset of 10
    x, y = create_training_data_from_csv(EKUSH_DIR + "female/chars/csv/femaleCharacters.csv", 10)
    save_numpy_data(x, y, EKUSH_DIR + "female/chars/")

    # Create Ekush female modifier dataset pickle using offset of 0
    x, y = create_training_data_from_csv(EKUSH_DIR + "female/modifiers/csv/femaleModifiers.csv", 0)
    save_numpy_data(x, y, EKUSH_DIR + "female/modifiers/")

    # Create Ekush female compound character dataset pickle using offset of 60
    x, y = create_training_data_from_csv(EKUSH_DIR + "female/compound/csv/femaleCompound.csv", 60)
    save_numpy_data(x, y, EKUSH_DIR + "female/compound/")


def initialise_matrivasha():
    # Create Matri Vasha male dataset pickle
    x, y = create_training_data_from_images(MATRIVASHA_CLASS_NAMES, MATRIVASHA_DIR + "male/images/")
    save_numpy_data(x, y, MATRIVASHA_DIR + "male/")

    # Create Matri Vasha female dataset pickle
    x, y = create_training_data_from_images(MATRIVASHA_CLASS_NAMES, MATRIVASHA_DIR + "female/images/")
    save_numpy_data(x, y, MATRIVASHA_DIR + "female/")


if __name__ == "__main__":
    initialise_ekush()
    initialise_matrivasha()
