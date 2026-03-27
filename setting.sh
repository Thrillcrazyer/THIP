apt-get update && apt-get install -y git

source .env

git config --global user.name "Taek-Hyun Park"
git config --global user.email "pthpark1@gmail.com"


git clone https://$GITHUB_TOKEN@github.com/Thrillcrazyer/THIP.git
cd THIP

pip install gdown
gdown 1KMnlO-Y5f_xp_FGvrXiOzY0mcz6HckgK 
gdown 13VGdmv5VjsZUPKxlAsDaT4HGxTMyp9TJ

gdown 1ILUMmKVxxDL4F0S5EYRVKGIKQZJlT0Wj #qwen32
gdown 12t4kNBFEA0Ps8dN_JpFS49y3y8JgvQpa #qwen14
gdown 1H8FS4HGsAZWt3bpdaPkL06nlMIixbT6O #llama


mkdir -p eventlogs
mv DeepMath_eventlog.csv eventlogs/

pip install -r requirements.txt

wandb login --verify  $WANDB_API_KEY
hf auth login --token $HF_TOKEN --add-to-git-credential

echo "✅ Environment setup complete!"



