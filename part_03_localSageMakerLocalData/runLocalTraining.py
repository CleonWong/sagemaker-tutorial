import jsonref

from sagemaker.tensorflow import TensorFlow

# ----------

aws_config = jsonref.load(open("./config/awsConfig.json"))

tf_estimator = TensorFlow(
    entry_point="train_cnn.py",
    role=aws_config["iam_role_arn"],
    instance_count=1,  # Have to provide, even if training locally.
    instance_type="local",
    framework_version="2.4.1",
    py_version="py37",
    source_dir="./src",  # Directory of all the code and requirements.txt.
    hyperparameters={
        "input_shape_w": 28,
        "input_shape_h": 28,
        "input_shape_c": 1,
        "learning_rate": 0.0001,
        "epochs": 1,
        "batch_size": 500,
    },  # SageMaker will turn this dict into command line arguments.
)

# Calling the .fit method starts the SageMaker training job.
tf_estimator.fit(
    inputs={
        "training": "file://data",
        "validation": "file://data",
    }
)
