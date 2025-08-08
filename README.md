# Fraudulent Claims Detection on AWS SageMaker

Production-grade repository for fraud detection using AWS SageMaker with XGBoost, Clarify, and Pipelines.

## Contents

- `src/preprocess.py`  SageMaker Processing script: pulls CSV from S3, splits into train/validation/test, writes baseline and batch samples.
- `src/train.py`       SageMaker Script Mode trainer for XGBoost. If your notebook training was detected, it is embedded; otherwise a robust trainer is provided.
- `src/evaluate.py`    SageMaker Processing evaluation that generates model metrics JSON for the Model Registry.
- `src/generate_config.py`  Generates Clarify analysis config and uploads to S3 for bias and SHAP analysis.
- `src/sagemaker_pipeline.py`  Full pipeline: Processing → Training → Evaluation → Register → Clarify → Batch Transform.
- `src/inference.py`   Simple client to invoke a deployed endpoint.

- `scripts/run_pipeline.sh`   Upsert and run the SageMaker Pipeline.
- `scripts/deploy_model.sh`   Deploy an approved Model Package to an endpoint.

- `config/*.json` Params and instance types.
- `notebooks/capstone.ipynb` Original notebook for exploration and reference.
- `data/*.csv` Small local samples for quick tests.

## Prereqs

- AWS account with SageMaker Studio or IAM role for pipelines.
- `pip install -r requirements.txt`
- AWS credentials configured locally.

## Quick Start

1. Upload input CSV to S3 (or adjust `InputDataS3Uri` when starting the pipeline).
2. Run the pipeline:
   ```bash
   bash scripts/run_pipeline.sh
   ```
3. Approve the model in SageMaker Model Registry, then deploy:
   ```bash
   export MODEL_PACKAGE_ARN=arn:aws:sagemaker:...
   bash scripts/deploy_model.sh
   ```
4. Invoke:
   ```bash
   python -m src.inference --endpoint fraud-endpoint --csv data/batch_data.csv
   ```

## Notes

- This repo is structured for real-world usage. Extend the pipeline with data quality checks, drift monitors, and CI/CD as needed.
- The pipeline script uses generic instance types from `config/sagemaker_config.json`. Adjust per your budget/SLAs.