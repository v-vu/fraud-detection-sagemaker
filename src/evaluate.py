import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost
import datetime as dt
from sklearn.metrics import roc_curve,auc


if __name__ == "__main__":
    
    #Read Model Tar File
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = xgboost.Booster()
    model.load_model("xgboost-model")
       
    #Read Test Data using which we evaluate the model
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)
    
    #Run Predictions
    predictions = model.predict(X_test)
    
    #Evaluate Predictions
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = auc(fpr, tpr)
    # Use the template at https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    # so that it will show up in the registry
    report_dict = {
        "binary_classification_metrics": {
            "auc" : {
                "value" : auc_score,
                "standard_deviation" : "0"
            },
        }
    }   
    
    #Save Evaluation Report
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))