apt update && apt upgrade
apt install ranger htop fish
conda init fish && fish
set -Ux AWS_ENDPOINT_URL https://sin1.contabostorage.com
set -Ux S3_ENDPOINT_URL https://sin1.contabostorage.com

conda activate base && conda install -y python=3.12
pip install poetry

mkdir ~/dev ~/datasets ~/experiments && cd ~/dev
git clone https://github.com/citruscaesar/geovision && cd geovision
git --global user.name "Sambhav Chandra"
git --global user.email "vader.sam10@gmail.com"
pip install poetry && poetry shell
poetry install
pip install -e .
