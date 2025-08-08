#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import sagemaker, boto3, os
from sagemaker import ModelPackage
sess = sagemaker.Session()
region = sess.boto_region_name
role = sagemaker.get_execution_role()
mpg = os.environ.get("MODEL_PACKAGE_ARN")
if not mpg:
    raise SystemExit("Set MODEL_PACKAGE_ARN to the approved Model Package ARN")
model = ModelPackage(role=role, model_package_arn=mpg, sagemaker_session=sess)
predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=os.environ.get("ENDPOINT_NAME","fraud-endpoint"))
print("Deployed endpoint:", predictor.endpoint_name)
PY