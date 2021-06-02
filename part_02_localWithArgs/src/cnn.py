import argparse
import os

import numpy as np

from modules.helpers import get_data, npy_to_tfdataset, build_cnn, train_cnn

# ----------


def main():

    # =================================
    #  Get arguments from command line
    # =================================

    parser = argparse.ArgumentParser(
        prog="MNIST classifier",
        description="Download MNIST data, build CNN, train and predict.",
        add_help=True,
    )

    parser.add_argument(
        "data_dir", help="Directory of folder to save MNIST data", type=str
    )

    parser.add_argument(
        "saved_model_dir",
        help="Directory of folder to save trained models",
        type=str,
    )

    parser.add_argument(
        "predictions_dir",
        help="Directory of folder to save model predictions",
        type=str,
    )

    parser.add_argument(
        "--input_shape",
        nargs="+",
        help=(
            "Shape of input images. E.g. (28, 28, 1) should be --input_shape 28 28 1"
        ),
        type=int,
        default=[28, 28, 1],
    )

    # Hyperparameters
    parser.add_argument(
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--learning_rate",
        help="Number of training epochs.",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--epochs",
        help="Learning rate for training.",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # ======
    #  Data
    # ======

    get_data(data_folder=args.data_dir)

    x_train = np.load(os.path.join(args.data_dir, "x_train.npy"))
    x_train = x_train / 255.0  # Normalise
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    train_ds = npy_to_tfdataset(X=x_train, y=y_train)

    x_test = np.load(os.path.join(args.data_dir, "x_test.npy"))
    x_test = x_test / 255.0  # Normalise
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    test_ds = npy_to_tfdataset(X=x_test)

    # for x, y in train_ds.take(1):
    #     print("Input:", x.shape)
    #     print("Target:", y.shape)

    # =============
    #  Build model
    # =============

    model = build_cnn(input_shape=args.input_shape)
    print(model.summary())

    # =============
    #  Train model
    # =============

    train_cnn(
        model=model,
        train_ds=train_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    # Save trained model locally
    model.save(os.path.join(args.saved_model_dir, "trained_cnn.h5"))

    # ============
    #  Model pred
    # ============

    test_ds = test_ds.batch(batch_size=args.batch_size)
    pred_prob_arr = model.predict(test_ds)
    pred_arr = pred_prob_arr.argmax(axis=1)

    np.save(os.path.join(args.predictions_dir, "pred.npy"), pred_arr)

    return


# ----------

if __name__ == "__main__":

    main()
