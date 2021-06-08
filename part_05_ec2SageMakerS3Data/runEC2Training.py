import jsonref

from sagemaker.tensorflow import TensorFlow

# ----------

aws_config = jsonref.load(open("./config/awsConfig.json"))

local_hyperparameters = {
    "input_shape_w": 28,
    "input_shape_h": 28,
    "input_shape_c": 1,
    "learning_rate": 0.0001,
    "epochs": 1,
    "batch_size": 500,
}

tf_estimator = TensorFlow(
    entry_point="train_cnn.py",
    source_dir="./src",  # Directory of all the code and requirements.txt.
    output_path=aws_config["s3_saved_model_folder"],  # Save the trained model to S3.
    role=aws_config["iam_role_arn"],
    instance_count=1,  # Have to provide, even if training locally.
    instance_type="ml.c4.xlarge",
    framework_version="2.4.1",
    py_version="py37",
    hyperparameters=local_hyperparameters,  # SageMaker will turn this dict into command line arguments.
)

# Calling the .fit method starts the SageMaker training job.
s3_inputs = {
    "training": aws_config["s3_data_folder"],
    "validation": aws_config["s3_data_folder"],
}
tf_estimator.fit(inputs=s3_inputs)
