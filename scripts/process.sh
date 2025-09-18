# Download dataset
python utils/download_DeepMath.py

# Activity Extraction
python preprocess.py

# Training Using DeepSeek-R1-Distill
bash scripts/train.sh

# Evaluation
bash scripts/eval.sh
