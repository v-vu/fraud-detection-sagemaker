import argparse
import boto3
import json
import pandas as pd

def invoke(endpoint, csv_path):
    runtime = boto3.client("sagemaker-runtime")
    payload = pd.read_csv(csv_path, header=None).to_csv(index=False, header=False)
    resp = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="text/csv",
        Body=payload
    )
    result = resp["Body"].read().decode("utf-8")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    print(invoke(args.endpoint, args.csv))