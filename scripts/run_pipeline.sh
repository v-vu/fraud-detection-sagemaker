#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from src.sagemaker_pipeline import get_pipeline
p = get_pipeline()
print("Creating/updating pipeline:", p.name)
p.upsert()
execution = p.start()
print("Started:", execution.arn)
print("Waiting for completion...")
execution.wait()
print("Pipeline execution completed:", execution.describe()['PipelineExecutionStatus'])
PY