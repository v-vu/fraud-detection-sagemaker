import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
import boto3
import argparse
import logging
import pathlib

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    
    #Read Data
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    
    fn = f"{base_dir}/data/claims_customer.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=0,
    )
    os.unlink(fn)
    
    # Split in Train, Test and Validation Datasets
    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(df))
    train, validation, test = np.split(
        df.sample(frac=1, random_state=1729),
        [int(0.7 * len(df)), int(0.9 * len(df))],
    )
    
    # Save the Dataframes as csv files
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    # Make a copy of the data frame and remove one column
    base_data = df.copy()
    base_data.pop("fraud")
    
    # For analysis config file generation for Clarify in generate_config.py
    baseline_sample = base_data.sample(frac=0.0002)
    baseline_sample.to_csv(f"{base_dir}/baseline/baseline.csv", header=False, index=False)
    
    # To support batch transform step
    batch_sample = base_data.sample(frac=0.2)
    batch_sample.to_csv(f"{base_dir}/batch/batch.csv", header=False, index=False)
    