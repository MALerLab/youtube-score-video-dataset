#!/bin/bash
CWD=$(pwd)
DATSET_DIR=/path/to/your/dataset

pipenv run python -m ytsv --dataset-dir $DATSET_DIR --metadata-path $CWD/metadata/ytsv_metadata.csv --checkpoint-dir $CWD/checkpoints --target-height 18 --device cuda:0
