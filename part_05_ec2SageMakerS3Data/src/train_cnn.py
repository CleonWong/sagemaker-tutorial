import argparse
import os

import numpy as np

from modules.helpers import get_data, npy_to_tfdataset, build_cnn, train_cnn, save_model

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

    # Set arguments from SageMaker:
    parser.add_argument(
        "--training",
        # help="Directory of folder to save MNIST data",
        help="Directory of folder for training data.",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
    )

    parser.add_argument(
        "--validation",
        help="Directory of folder for validation data.",
        type=str,
        default=os.environ["SM_CHANNEL_VALIDATION"],
    )

    parser.add_argument(
        "--model_dir",
        help="Directory of folder to save trained models",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )

    parser.add_argument(
        "--gpu_count",
        help="Number of GPUs available.",
        type=int,
        default=os.environ["SM_NUM_GPUS"],
    )

    # My own optional arguments:
    parser.add_argument(
        "--input_shape_w",
        help="Width of input images.",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--input_shape_h",
        help="Height of input images.",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--input_shape_c",
        help="Number of channels in input images. RGB = 3, black and white = 1.",
        type=int,
        default=1,
    )

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

    # Print args
    for arg in vars(args):
        print(arg, getattr(args, arg))

    input_shape_w = args.input_shape_w
    input_shape_h = args.input_shape_h
    input_shape_c = args.input_shape_c
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation
    gpu_count = args.gpu_count

    # ======
    #  Data
    # ======

    x_train = np.load(os.path.join(training_dir, "x_train.npy"))
    x_train = x_train / 255.0  # Normalise
    y_train = np.load(os.path.join(training_dir, "y_train.npy"))
    train_ds = npy_to_tfdataset(X=x_train, y=y_train)

    # =============
    #  Build model
    # =============

    input_shape = (input_shape_w, input_shape_h, input_shape_c)
    model = build_cnn(input_shape=input_shape)
    print(model.summary())

    # =============
    #  Train model
    # =============

    train_cnn(
        model=model,
        train_ds=train_ds,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # Save trained model locally
    # save_model(model=model, dir=os.environ["SM_MODEL_DIR"])
    save_model(model=model, dir=model_dir)

    return


# ----------

if __name__ == "__main__":

    main()
