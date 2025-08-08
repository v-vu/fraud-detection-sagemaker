import os
import sagemaker
import boto3
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.clarify_check_step import ClarifyCheckStep, DataConfig, ModelConfig, ModelPredictedLabelConfig, BiasConfig, CheckJobConfig
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep

def get_pipeline(region=None, role=None, default_bucket=None, pipeline_name="fraud-pipeline"):
    session = sagemaker.session.Session()
    region = region or session.boto_region_name
    role = role or sagemaker.get_execution_role()
    default_bucket = default_bucket or session.default_bucket()
    p_session = PipelineSession()

    # Parameters
    input_s3 = ParameterString(name="InputDataS3Uri", default_value=f"s3://{default_bucket}/input/claims_customer.csv")
    clar_output = ParameterString(name="ClarifyOutputS3Uri", default_value=f"s3://{default_bucket}/clarify-output/")
    instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)

    cache = CacheConfig(enable_caching=True, expire_after="30d")

    # Processing: preprocessing
    processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.2-1"),
        command=["python3"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=p_session,
    )
    step_process = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[ProcessingInput(source=input_s3, destination="/opt/ml/processing/input/claims_customer.csv")],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/train", output_name="train"),
            ProcessingOutput(source="/opt/ml/processing/validation", output_name="validation"),
            ProcessingOutput(source="/opt/ml/processing/test", output_name="test"),
            ProcessingOutput(source="/opt/ml/processing/batch", output_name="batch"),
            ProcessingOutput(source="/opt/ml/processing/baseline", output_name="baseline"),
        ],
        code="src/preprocess.py",
        cache_config=cache,
    )

    # Training: XGBoost
    xgb_image = sagemaker.image_uris.retrieve("xgboost", region=region, version="1.7-1")
    xgb = XGBoost(
        entry_point="train.py",
        source_dir="src",
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="1.7-1",
        py_version="py3",
        sagemaker_session=p_session,
        hyperparameters={"max_depth": 5, "eta": 0.2, "num_round": 200, "objective": "binary:logistic", "eval_metric": "auc"},
        image_uri=xgb_image,
    )
    step_train = TrainingStep(
        name="TrainModel",
        estimator=xgb,
        inputs={
            "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv"),
            "validation": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri, content_type="text/csv"),
        },
        cache_config=cache,
    )

    # Evaluation
    eval_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.2-1"),
        command=["python3"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=p_session,
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/evaluation", output_name="evaluation")],
        code="src/evaluate.py",
        cache_config=cache,
    )

    # Register Model with metrics
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/evaluation.json",
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=xgb,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_metrics=model_metrics,
        description="Fraud XGBoost Model",
        model_package_group_name="FraudModelPackageGroup",
    )

    # Clarify Bias and SHAP
    clarify_config = ClarifyCheckStep(
        name="ClarifyBiasAndExplainability",
        clarify_check_config=CheckJobConfig(
            role=role,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            sagemaker_session=p_session,
        ),
        data_config=DataConfig(
            s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            s3_output_path=clar_output,
            label="fraud",
            headers=None,
            dataset_type="text/csv",
        ),
        model_config=ModelConfig(
            model_name=None,
            instance_count=1,
            instance_type="ml.m5.xlarge",
            accept_type="text/csv",
        ),
        # This step can be updated to use generated analysis_config.json if preferred
    )

    # Batch Transform (for batch scoring)
    transformer = Transformer(
        model_name=step_register.properties.ModelPackageArn,
        instance_type="ml.m5.large",
        instance_count=1,
        strategy="SingleRecord",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}/batch-output/",
        sagemaker_session=p_session,
    )
    step_transform = TransformStep(
        name="BatchTransform",
        transformer=transformer,
        inputs=step_process.properties.ProcessingOutputConfig.Outputs["batch"].S3Output.S3Uri,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_s3, clar_output, instance_type, instance_count],
        steps=[step_process, step_train, step_eval, step_register, clarify_config, step_transform],
        sagemaker_session=p_session,
    )
    return pipeline