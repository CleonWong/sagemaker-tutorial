import jsonref
from sagemaker.tensorflow import TensorFlow

# ----------

aws_config = jsonref.load(open("../config/awsConfig.json"))

# ----------

tf_estimator = Tensorflow(
    entry_point="cnn.py",
    role=aws_config["iam_role_arn"],
)
