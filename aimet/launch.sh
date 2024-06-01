#!/bin/bash
pip install -r requirements.txt
pip install --upgrade transformers==4.23.1 --no-deps
pip install --upgrade onnx==1.13.0 --no-deps
pip install --upgrade onnxsim==0.4.10 --no-deps
pip install notebook==6.1.5 tokenizers==0.11.1
echo "Launch Jupyter notebook"
jupyter notebook --ip=* --no-browser --allow-root
