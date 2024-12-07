apt update && apt upgrade -y
apt install btop ranger fish awscli -y
conda init bash && conda init fish
conda create -n dev python=3.12
fish
fish_vi_key_bindings