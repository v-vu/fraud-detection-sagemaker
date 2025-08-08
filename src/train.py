# Auto-extracted training code candidates from capstone.ipynb
# Review and tailor as needed.

if __name__ == "__main__":
    #required-libraries

    import boto3
    import datetime as datetime
    import io
    import IPython
    import json
    import math
    import matplotlib.pyplot as plt  # visualization
    import numpy as np
    import pandas as pd
    import pathlib
    import re
    import sagemaker
    import seaborn as sns  # visualization
    import statistics
    import string
    import sys
    import time
    import zipfile

    from imblearn.over_sampling import SMOTE

    from sagemaker import clarify
    from sagemaker import get_execution_role
    from sagemaker.analytics import ExperimentAnalytics
    from sagemaker.dataset_definition.inputs import AthenaDatasetDefinition, DatasetDefinition, RedshiftDatasetDefinition
    from sagemaker.debugger import CollectionConfig, DebuggerHookConfig, FrameworkProfile, ProfilerConfig, ProfilerRule, Rule, rule_configs
    from sagemaker.estimator import Estimator
    from sagemaker.experiments.run import Run, load_run
    from sagemaker.feature_store.feature_definition import FeatureDefinition
    from sagemaker.feature_store.feature_definition import FeatureTypeEnum
    from sagemaker.feature_store.feature_group import FeatureGroup
    from sagemaker.inputs import CreateModelInput
    from sagemaker.inputs import TrainingInput
    from sagemaker.inputs import TransformInput
    from sagemaker.model import Model
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.network import NetworkConfig
    from sagemaker.processing import FeatureStoreOutput
    from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput, ScriptProcessor
    from sagemaker.pytorch.estimator import PyTorch
    from sagemaker.s3 import S3Uploader
    from sagemaker.session import Session
    from sagemaker.sklearn.processing import SKLearnProcessor
    from sagemaker.transformer import Transformer
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
    from sagemaker.workflow.condition_step import ConditionStep, JsonGet
    from sagemaker.workflow.conditions import ConditionGreaterThan
    from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.step_collections import RegisterModel
    from sagemaker.workflow.steps import CreateModelStep
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    from sagemaker.workflow.steps import TransformStep
    from sagemaker.workflow.steps import TuningStep
    from sagemaker.xgboost.estimator import XGBoost
    from sklearn.model_selection import train_test_split

    # ===== Next Training Candidate Cell =====

    #add_your_task_3_3_code_here
    from sagemaker import image_uris
    container = image_uris.retrieve(framework='xgboost',region=boto3.Session().region_name,version='1.5-1')

    # initialize hyperparameters
    eta=0.2
    gamma=4
    max_depth=5
    min_child_weight=6
    num_round=800
    objective='binary:logistic'
    subsample=0.8
    verbosity=0

    hyperparameters = {
            "max_depth":max_depth,
            "eta":eta,
            "gamma":gamma,
            "min_child_weight":min_child_weight,
            "subsample":subsample,
            "verbosity":verbosity,
            "objective":objective,
            "num_round":num_round
    }

    # Set up the estimator
    xgb = sagemaker.estimator.Estimator(
        container,
        role, 
        instance_count=1, 
        instance_type='ml.m5.xlarge',
        output_path='s3://{}/{}/output'.format(bucket, prefix),
        sagemaker_session=sagemaker_session,
        EnableSageMakerMetricsTimeSeries=True,
        hyperparameters=hyperparameters,
        tags = run_tags
    )


    #Run the training job link to Experiment.
    with Run(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=run_tags,
        sagemaker_session=sagemaker_session,
    ) as run:

        run.log_parameters({
                            "eta": eta, 
                            "gamma": gamma, 
                            "max_depth": max_depth,
                            "min_child_weight": min_child_weight,
                            "num_round": num_round,
                            "objective": objective,
                            "subsample": subsample,
                            "verbosity": verbosity
                           })
    
    #    you may also specify metrics to log
    #    run.log_metric(name="", value=x)

    # Train the model associating the training run with the current "experiment"
        xgb.fit(
            inputs = data_inputs
        ) 


    # ===== Next Training Candidate Cell =====

    #add_your_task_3_4_code_here
    #tune-model
    from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

    # Setup the hyperparameter ranges
    hyperparameter_ranges = {
        'alpha': ContinuousParameter(0, 2),
        'eta': ContinuousParameter(0, 1),
        'max_depth': IntegerParameter(1, 10),
        'min_child_weight': ContinuousParameter(1, 10),
        'num_round': IntegerParameter(100, 1000)
    }
    # Define the target metric and the objective type (max/min)
    objective_metric_name = 'validation:auc'
    objective_type='Maximize'
    # Define the HyperparameterTuner
    tuner = HyperparameterTuner(
        estimator = xgb,
        objective_metric_name = objective_metric_name,
        hyperparameter_ranges = hyperparameter_ranges,
        objective_type = objective_type,
        max_jobs=12,
        max_parallel_jobs=4,
        early_stopping_type='Auto',
    )

    with load_run(sagemaker_session=sagemaker_session, experiment_name=experiment_name, run_name=run_name) as run:
    # Tune the model
        tuner.fit(
            inputs = data_inputs,
            job_name = experiment_name,
        )
    


    # ===== Next Training Candidate Cell =====


    from sagemaker import image_uris
    #train-model
    # Retrieve the container image
    container = sagemaker.image_uris.retrieve(
        region=boto3.Session().region_name, 
        framework="xgboost", 
        version="1.5-1"
    )

    # Set the hyperparameters
    eta=0.2
    gamma=4
    max_depth=4
    min_child_weight=6
    num_round=800
    objective='binary:logistic'
    subsample=0.8

    hyperparameters = {
        "eta":eta,
        "gamma":gamma,
        "max_depth":max_depth,
        "min_child_weight":min_child_weight,
        "num_round":num_round,
        "objective":objective,
        "subsample":subsample
    }

    # Set up the estimator
    xgb = sagemaker.estimator.Estimator(
        container,
        role,    
        instance_count=1, 
        instance_type="ml.m5.4xlarge",
        output_path="s3://{}/{}/output".format(bucket, prefix),
        sagemaker_session=sagemaker_session,
        max_run=1800,
        hyperparameters=hyperparameters,
        tags = run_tags
    )

    with Run(
        experiment_name=capstone_experiment_name,
        run_name=capstone_run_name,
        sagemaker_session=sagemaker_session,
    ) as run:
        run.log_parameter("eta", eta)
        run.log_parameter("gamma", gamma)
        run.log_parameter("max_depth", max_depth)
        run.log_parameter("min_child_weight", min_child_weight)
        run.log_parameter("objective", objective)
        run.log_parameter("subsample", subsample)
        run.log_parameter("num_round", num_round)

    # Train the model associating the training run with the current "experiment"
        xgb.fit(
            inputs = data_inputs
        )        

    # ===== Next Training Candidate Cell =====

    #enable-debugger
    # Retrieve the container image
    container = sagemaker.image_uris.retrieve(
        region=boto3.Session().region_name, 
        framework="xgboost", 
        version="1.5-1"
    )

    # Set the hyperparameters
    eta=0.2
    gamma=4
    max_depth=4
    min_child_weight=6
    num_round=300
    objective='binary:logistic'
    subsample=0.7
        
    hyperparameters = {
            "eta":eta,
            "gamma":gamma,
            "max_depth":max_depth,
            "min_child_weight":min_child_weight,
            "num_round":num_round,
            "objective":objective,
            "subsample":subsample
    }

    # Set up the estimator
    xgb = sagemaker.estimator.Estimator(
        container,
        role, 
        base_job_name=base_job_name,
        instance_count=1, 
        instance_type="ml.m5.4xlarge",
        output_path="s3://{}/{}/output".format(bucket, prefix),
        sagemaker_session=sagemaker_session,
        max_run=1800,
        hyperparameters=hyperparameters,
        tags = run_tags,

        #Set the Debugger Hook Config
        debugger_hook_config=DebuggerHookConfig(
            s3_output_path=bucket_path,  # Required
            collection_configs=[
                CollectionConfig(name="metrics", parameters={"save_interval": str(save_interval)}),
                CollectionConfig(name="feature_importance", parameters={"save_interval": str(save_interval)},),
                CollectionConfig(name="full_shap", parameters={"save_interval": str(save_interval)}),
                CollectionConfig(name="average_shap", parameters={"save_interval": str(save_interval)}),
            ],
            ),
            #Set the Debugger Profiler Configuration
            profiler_config = ProfilerConfig(
                system_monitor_interval_millis=500,
                framework_profile_params=FrameworkProfile()
        ),
            #Configure the Debugger Rule Object
            rules = [
                ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
                Rule.sagemaker(rule_configs.create_xgboost_report()),  
                Rule.sagemaker(rule_configs.overfit()),
                Rule.sagemaker(rule_configs.overtraining()),
                Rule.sagemaker(rule_configs.loss_not_decreasing(),
                    rule_parameters={
                        "collection_names": "metrics",
                        "num_steps": str(save_interval * 2),
                    }
                )
        ]
    )
    with Run(
        experiment_name=capstone_experiment_name,
        run_name=capstone_run_name,
        sagemaker_session=sagemaker_session,
    ) as run:
        run.log_parameter("eta", eta)
        run.log_parameter("gamma", gamma)
        run.log_parameter("max_depth", max_depth)
        run.log_parameter("min_child_weight", min_child_weight)
        run.log_parameter("objective", objective)
        run.log_parameter("subsample", subsample)
        run.log_parameter("num_round", num_round)
    # Train the model
    xgb.fit(
        inputs = data_inputs
    ) 

    # ===== Next Training Candidate Cell =====

    #create-estimator
    hyperparameters= {
        "max_depth": "4",
        "eta": "0.2",
        "gamma": "4",
        "min_child_weight": "6",
        "subsample": "0.7",
        "objective": "binary:logistic",
        "num_round": "300",
    }

    xgb_retrained = sagemaker.estimator.Estimator(
        container,
        role, 
        instance_count=1, 
        instance_type="ml.m5.xlarge",
        output_path="s3://{}/{}/output".format(bucket, prefix),
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters
    )

    # ===== Next Training Candidate Cell =====

    #run-pipeline
    # Set the variables
    model_name = "Auto-model"
    sklearn_processor_version="0.23-1"
    model_package_group_name="AutoModelPackageGroup"
    pipeline_name= "AutoModelSMPipeline"
    clarify_image = sagemaker.image_uris.retrieve(framework='sklearn',version=sklearn_processor_version,region=region)

    # Upload files to the default S3 bucket
    s3_client.put_object(Bucket=bucket,Key='data/')
    s3_client.put_object(Bucket=bucket,Key='input/code/')
    s3_client.upload_file(Filename="data/batch_data.csv", Bucket=bucket, Key="data/batch_data.csv")  #If you edit this, make sure to also edit the headers listed in generate_config to match your column names.
    s3_client.upload_file(Filename="data/claims_customer.csv", Bucket=bucket, Key="data/claims_customer.csv")  #If you edit this, make sure to also edit the headers listed in generate_config to match your column names.
    s3_client.upload_file(Filename="pipelines/evaluate.py", Bucket=bucket, Key="input/code/evaluate.py")
    s3_client.upload_file(Filename="pipelines/generate_config.py", Bucket=bucket, Key="input/code/generate_config.py")
    s3_client.upload_file(Filename="pipelines/preprocess.py", Bucket=bucket, Key="input/code/preprocess.py")

    # Configure important settings. Change the input_data if you want to
    # use a file other than the claims_customer.csv and batch_data.csv files.
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1
    )
    processing_instance_type = ParameterString(
            name="ProcessingInstanceType",
            default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
            name="TrainingInstanceType",
            default_value="ml.m5.xlarge"
    )
    input_data = ParameterString(
            name="InputData",
            default_value="s3://{}/data/claims_customer.csv".format(bucket), 
    )
    batch_data = ParameterString(
            name="BatchData",
            default_value="s3://{}/data/batch_data.csv".format(bucket),
    )

    # Run a scikit-learn script to do data processing on SageMaker using 
    # using the SKLearnProcessor class
    sklearn_processor = SKLearnProcessor(
            framework_version=sklearn_processor_version,
            instance_type=processing_instance_type.default_value, 
            instance_count=processing_instance_count,
            sagemaker_session=sagemaker_session,
            role=role,
    )

    # Configure the processing step to pull in the input_data
    step_process = ProcessingStep(
            name="AutoModelProcess",
            processor=sklearn_processor,
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/train",\
                                 destination=f"s3://{bucket}/output/train" ),
                ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation",\
                                destination=f"s3://{bucket}/output/validation"),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test",\
                                destination=f"s3://{bucket}/output/test"),
                ProcessingOutput(output_name="batch", source="/opt/ml/processing/batch",\
                                destination=f"s3://{bucket}/data/batch"),
                ProcessingOutput(output_name="baseline", source="/opt/ml/processing/baseline",\
                                destination=f"s3://{bucket}/input/baseline")
            ],
            code=f"s3://{bucket}/input/code/preprocess.py",
            job_arguments=["--input-data", input_data],
    )

    # Set up the model path, image uri, and hyperparameters for the estimator
    model_path = f"s3://{bucket}/output"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type=training_instance_type.default_value,
    )

    fixed_hyperparameters = {
        "eval_metric":"auc",
        "objective":"binary:logistic",
        "num_round":"100",
        "rate_drop":"0.3",
        "tweedie_variance_power":"1.4"
    }

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        hyperparameters=fixed_hyperparameters,
        output_path=model_path,
        base_job_name=f"auto-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    # Set the hyperparameter ranges for the tuning step and configure the tuning step
    hyperparameter_ranges = {
        "eta": ContinuousParameter(0, 1),
        "min_child_weight": ContinuousParameter(1, 10),
        "alpha": ContinuousParameter(0, 2),
        "max_depth": IntegerParameter(1, 4),
    }
    objective_metric_name = "validation:auc"

    step_tuning = TuningStep(
        name = "AutoHyperParameterTuning",
        tuner = HyperparameterTuner(xgb_train, objective_metric_name, hyperparameter_ranges, max_jobs=2, max_parallel_jobs=2),
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Configure the processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="script-auto-eval",
        role=role,
        sagemaker_session=sagemaker_session,
    )

    evaluation_report = PropertyFile(
        name="AutoEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="AutoEvalBestModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=bucket,prefix="output"),
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",\
                                destination=f"s3://{bucket}/output/evaluation"),
        ],
        code=f"s3://{bucket}/input/code/evaluate.py",
        property_files=[evaluation_report],
    )

    # Configure model creation
    model = Model(
        image_uri=image_uri,        
        model_data=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=bucket,prefix="output"),
        name=model_name,
        sagemaker_session=sagemaker_session,
        role=role,
    )

    inputs = CreateModelInput(
        instance_type="ml.m5.large",
        accelerator_type="ml.inf1.xlarge",
    )

    step_create_model = CreateModelStep(
        name="AutoCreateModel",
        model=model,
        inputs=inputs,
    )

    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri=clarify_image,
        role=role,
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=sagemaker_session,
    )

    bias_report_output_path = f"s3://{bucket}/clarify-output/bias"
    clarify_instance_type = 'ml.m5.xlarge'
    step_config_file = ProcessingStep(
        name="AutoModelConfigFile",
        processor=script_processor,
        code=f"s3://{bucket}/input/code/generate_config.py",
        job_arguments=["--modelname",step_create_model.properties.ModelName,"--bias-report-output-path",bias_report_output_path,"--clarify-instance-type",clarify_instance_type,\
                      "--default-bucket",bucket,"--num-baseline-samples","50","--instance-count","1"],
        depends_on= [step_create_model.name]
    )

    # Configure the step to perform a batch transform job
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        assemble_with="Line",
        accept="text/csv",    
        output_path=f"s3://{bucket}/AutoTransform"
    )

    step_transform = TransformStep(
        name="AutoTransform",
        transformer=transformer,
        inputs=TransformInput(data=batch_data,content_type="text/csv",join_source="Input",split_type="Line")
    )

    # Configure the SageMaker Clarify processing step
    analysis_config_path = f"s3://{bucket}/clarify-output/bias/analysis_config.json"

    data_config = sagemaker.clarify.DataConfig(
        s3_data_input_path=f's3://{bucket}/output/train/train.csv', 
        s3_output_path=bias_report_output_path,
        label=0,
        headers=list(pd.read_csv("./data/claims_customer.csv", index_col=None).columns), #If you edit this, make sure to also edit the headers listed in generate_config to match your column names.
        dataset_type="text/csv",
    )

    clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type=clarify_instance_type,
        sagemaker_session=sagemaker_session,
    )

    config_input = ProcessingInput(
        input_name="analysis_config",
        source=analysis_config_path,
        destination="/opt/ml/processing/input/analysis_config",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_compression_type="None",
    )

    data_input = ProcessingInput(
        input_name="dataset",
        source=data_config.s3_data_input_path,
        destination="/opt/ml/processing/input/data",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type=data_config.s3_data_distribution_type,
        s3_compression_type=data_config.s3_compression_type,
    )

    result_output = ProcessingOutput(
        source="/opt/ml/processing/output",
        destination=data_config.s3_output_path,
        output_name="analysis_result",
        s3_upload_mode="EndOfJob",
    )

    step_clarify = ProcessingStep(
        name="ClarifyProcessingStep",
        processor=clarify_processor,
        inputs= [data_input, config_input],
        outputs=[result_output],
        depends_on = [step_config_file.name]
    )

    # Configure the model registration step
    model_statistics = MetricsSource(
        s3_uri="s3://{}/output/evaluation/evaluation.json".format(bucket),
        content_type="application/json"
    )
    explainability = MetricsSource(
        s3_uri="s3://{}/clarify-output/bias/analysis.json".format(bucket),
        content_type="application/json"
    )

    bias = MetricsSource(
        s3_uri="s3://{}/clarify-output/bias/analysis.json".format(bucket),
        content_type="application/json"
    ) 

    model_metrics = ModelMetrics(
        model_statistics=model_statistics,
        explainability=explainability,
        bias=bias
    )

    step_register = RegisterModel(
        name="RegisterAutoModel",
        estimator=xgb_train,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0,s3_bucket=bucket,prefix="output"),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
    )

    # Create the model evaluation step
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.auc.value"
        ),
        right=0.75,
    )

    step_cond = ConditionStep(
        name="CheckAUCScoreAutoEvaluation",
        conditions=[cond_lte],
        if_steps=[step_create_model,step_config_file,step_transform,step_clarify,step_register],
        else_steps=[],
    )

    # Define the pipeline
    def get_pipeline(
        region,
        role=None,
        default_bucket=None,
        model_package_group_name="AutoModelPackageGroup",
        pipeline_name="AutoModelPipeline",
        base_prefix = None,
        custom_image_uri = None,
        sklearn_processor_version=None
        ):
        """Gets a SageMaker ML Pipeline instance working with auto data.
        Args:
            region: AWS region to create and run the pipeline.
            role: IAM role to create and run steps and pipeline.
            default_bucket: the bucket to use for storing the artifacts
        Returns:
            an instance of a pipeline
        """

        # pipeline instance
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[
                processing_instance_type,
                processing_instance_count,
                training_instance_type,
                input_data,
                batch_data,
            ],
            steps=[step_process,step_tuning,step_eval,step_cond],
            sagemaker_session=sagemaker_session
        )
        return pipeline


    # Create the pipeline
    pipeline = get_pipeline(
        region = region,
        role=role,
        default_bucket=bucket,
        model_package_group_name=model_package_group_name,
        pipeline_name=pipeline_name,
        custom_image_uri=clarify_image,
        sklearn_processor_version=sklearn_processor_version
    )

    pipeline.upsert(role_arn=role)

    # Run the pipeline
    RunPipeline = pipeline.start()
