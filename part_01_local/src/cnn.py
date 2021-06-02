import numpy as np

from modules.helpers import get_data, npy_to_tfdataset, build_cnn, train_cnn

# ----------


def main():

    # ======
    #  Data
    # ======

    get_data(data_folder="../data")

    x_train = np.load("../data/x_train.npy")
    x_train = x_train / 255.0  # Normalise
    y_train = np.load("../data/y_train.npy")
    train_ds = npy_to_tfdataset(X=x_train, y=y_train)

    x_test = np.load("../data/x_test.npy")
    x_test = x_test / 255.0  # Normalise
    y_test = np.load("../data/y_test.npy")
    test_ds = npy_to_tfdataset(X=x_test)

    # print(list(train_ds.as_numpy_iterator())[0][0].shape)

    for x, y in train_ds.take(1):
        print("Input:", x.shape)
        print("Target:", y.shape)

    # =============
    #  Build model
    # =============

    model = build_cnn(input_shape=(28, 28, 1))
    print(model.summary())

    # =============
    #  Train model
    # =============

    train_cnn(
        model=model, train_ds=train_ds, batch_size=500, epochs=10, learning_rate=0.0001
    )

    # Save trained model locally
    model.save("../output/saved_model/trained_cnn.h5")

    # ============
    #  Model pred
    # ============

    test_ds = test_ds.batch(batch_size=500)
    pred_prob_arr = model.predict(test_ds)
    pred_arr = pred_prob_arr.argmax(axis=1)

    np.save("../output/predictions/pred.npy", pred_arr)

    return


# ----------

if __name__ == "__main__":

    main()
