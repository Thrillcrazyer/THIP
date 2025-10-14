apt-get update && apt-get install -y git

source .env

git config --global user.name "Taek-Hyun Park"
git config --global user.email "pthpark1@gmail.com"


git clone https://$GITHUB_TOKEN@github.com/Thrillcrazyer/R.i.P_ReasoningisProcess.git
cd R.i.P_ReasoningisProcess

pip install gdown
gdown 1KMnlO-Y5f_xp_FGvrXiOzY0mcz6HckgK
gdown 13VGdmv5VjsZUPKxlAsDaT4HGxTMyp9TJ

mkdir -p eventlogs
mv DeepMath_eventlog.csv eventlogs/

pip install -r requirements.txt

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_API_KEY

echo "✅ Environment setup complete!"



