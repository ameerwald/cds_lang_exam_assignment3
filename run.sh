#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/model_train.py
python3 src/generate_text.py

# deactive the venv
deactivate