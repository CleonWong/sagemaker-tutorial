import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# ----------


def get_data(data_folder):

    """
    This function loads the MNIST data from the Tensorflow library and saves it
    in the specified outout folder. The numpy arrays are by default called
    "x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy".

    Parameters
    ----------
    data_folder : str or Path
        The folder to save the MNIST data to.

    Returns
    -------
    True if successful else False.
    """

    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        new_x_train_list = []
        for i, _ in enumerate(x_train):
            arr = np.reshape(x_train[i], (28, 28, 1))
            new_x_train_list.append(arr)

        new_x_test_list = []
        for i, _ in enumerate(x_test):
            arr = np.reshape(x_test[i], (28, 28, 1))
            new_x_test_list.append(arr)

        new_x_train = np.array(new_x_train_list)
        new_x_test = np.array(new_x_test_list)

        np.save(os.path.join(data_folder, "x_train.npy"), new_x_train)
        np.save(os.path.join(data_folder, "y_train.npy"), y_train)
        np.save(os.path.join(data_folder, "x_test.npy"), new_x_test)
        np.save(os.path.join(data_folder, "y_test.npy"), y_test)

    except Exception as ex:
        print("ERROR: get_data().")
        print(ex)


def npy_to_tfdataset(X, y=None):

    """
    This function creates a Tensorflow dataset object from numpy arrays.

    Parameters
    ----------
    X : np.array
        Training images.
    y: np.array, default=None
        Corresponding labels.

    Returns
    -------
    [type]
        [description]
    """

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    return dataset


def build_cnn(input_shape=(28, 28, 1)):

    """
    This function builds the CNN using the specified `input_shape`.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of image in (image width, image height, channels)

    Returns
    -------
    model : tensorflow.python.keras.engine.sequential.Sequential
    """

    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape)
    )
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.softmax))

    return model


def train_cnn(model, train_ds, batch_size=50, epochs=10, learning_rate=0.0001):

    """
    This function compiles and fits a given model using the x_train and y_train
    data.

    Parameters
    ----------
    model : TF model object
        The CNN model to train.
    train_ds : Tensorflow Dataset object
        A Tensorflow Dataset object, output from npy_to_tfdataset().
    batch_size: int, default = 50
        Training batch size.
    epochs : int, default = 10
        Number of training epochs.
    learning_rate : float, default = 0.0001
        The learning rate.

    Returns
    -------
    """

    # Batch the dataset.
    train_ds = train_ds.batch(batch_size=batch_size)

    # list(train_ds.as_numpy_iterator())

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    model.fit(train_ds, epochs=epochs, verbose=1)

    return
