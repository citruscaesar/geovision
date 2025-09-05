#!/bin/bash
echo "performing system update and installing required packages"
apt update && apt -y upgrade
apt -y install btop neovim rsync fzf ripgrep bat fish
echo "cleaning up"
apt -y autoremove && apt -y autoclean

echo "creating project directories"
mkdir ~/dev ~/datasets ~/experiments ~/models
cd ~/dev && git clone https://github.com/citruscaesar/geovision.git 

echo "installing pixi"
curl -fsSL https://pixi.sh/install.sh | bash 
echo "alias bat=batcat" >> ~/.bashrc
echo 'if [ -f ~/.bashrc ]; then source ~/.bashrc; fi' >> ~/.bash_profile
source ~/.bashrc