git config --global user.name "Sambhav Chandra"
git config --global user.email "vader.sam10@gmail.com"
apt update && apt upgrade -y
apt install btop ranger fish awscli neovim -y
conda init bash 
exec "$BASH"
conda deactivate 
conda create -n dev  
conda activate dev
conda install -y python=3.12 poetry
#git clone --depth 1 https://github.com/citruscaesar/geovision