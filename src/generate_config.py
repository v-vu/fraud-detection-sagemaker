import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
import sagemaker
import argparse
import json
import boto3


if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname",type=str, required=True)
    parser.add_argument("--bias-report-output-path",type=str, required=True)
    parser.add_argument("--clarify-instance-type",type=str, required=True)
    parser.add_argument("--default-bucket",type=str, required=True)
    parser.add_argument("--instance-count",type=int, required=True)
    parser.add_argument("--num-baseline-samples",type=int, required=True)

    args = parser.parse_args()
    base_dir = "/opt/ml/processing"
    
    # Get the headers for the dataset
    header_list = ['fraud', 'policy_id', 'incident_type_theft', 'policy_state_ca',
       'policy_deductable', 'num_witnesses', 'policy_state_or',
       'incident_month', 'customer_gender_female', 'num_insurers_past_5_years',
       'customer_gender_male', 'total_claim_amount',
       'authorities_contacted_police', 'incident_day', 'collision_type_side',
       'customer_age', 'customer_education', 'driver_relationship_child',
       'driver_relationship_spouse', 'injury_claim', 'incident_dow',
       'collision_type_na', 'incident_severity', 'driver_relationship_self',
       'num_claims_past_year', 'months_as_customer', 'auto_year',
       'num_vehicles_involved', 'policy_state_nv',
       'authorities_contacted_ambulance', 'num_injuries', 'policy_liability',
       'police_report_available', 'driver_relationship_na',
       'incident_type_breakin', 'authorities_contacted_none',
       'authorities_contacted_fire', 'driver_relationship_other',
       'policy_state_id', 'incident_hour', 'vehicle_claim',
       'incident_type_collision', 'policy_annual_premium', 'policy_state_az',
       'policy_state_wa', 'collision_type_rear', 'collision_type_front']
    
    # Read the train dataset
    data_config = sagemaker.clarify.DataConfig(
        s3_data_input_path=f's3://{args.default_bucket}/output/train/train.csv',
        s3_output_path=args.bias_report_output_path,
        label="fraud",
        headers=header_list,
        dataset_type="text/csv",
    )
    model_config = sagemaker.clarify.ModelConfig(
        model_name=args.modelname,
        instance_type=args.clarify_instance_type,
        instance_count=int(args.instance_count),
        accept_type="text/csv",
    )
    model_predicted_label_config = sagemaker.clarify.ModelPredictedLabelConfig(probability_threshold=0.5)
    bias_config = sagemaker.clarify.BiasConfig(
        label_values_or_threshold=[1],
        facet_name="customer_gender_female",
        facet_values_or_threshold=[0],
        group_name="customer_age",
    )
    analysis_config = data_config.get_config()
    analysis_config.update(bias_config.get_config())
    analysis_config["predictor"] = model_config.get_predictor_config()
    if model_predicted_label_config:
        (
            probability_threshold,
            predictor_config,
        ) = model_predicted_label_config.get_predictor_config()
        if predictor_config:
            analysis_config["predictor"].update(predictor_config)
        if probability_threshold is not None:
            analysis_config["probability_threshold"] = probability_threshold
    analysis_config["methods"] = {
        "pre_training_bias": {"methods": 'all'},
        "post_training_bias": {"methods": 'all'},
        "shap":{"baseline":f"s3://{args.default_bucket}/input/baseline/baseline.csv",
               "num_samples":int(args.num_baseline_samples),
               "agg_method":"mean_abs",
               "save_local_shap_values":True},
    }
    analysis_config["methods"]["report"] = {"name": "report", "title": "Analysis Report"}
    analysis_config_file = os.path.join(f"{base_dir}/analysis_config.json")
    
    # Save the Config File to S3
    with open(analysis_config_file, "w") as f:
        json.dump(analysis_config, f)
    s3_client = boto3.resource('s3')
    s3_client.Bucket(args.default_bucket).upload_file(analysis_config_file,"clarify-output/bias/analysis_config.json")
    