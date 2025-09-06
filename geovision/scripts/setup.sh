#!/bin/bash
echo "performing system update and installing required packages"
apt update && apt -y upgrade
apt -y install btop neovim rsync fzf ripgrep bat fish
echo "cleaning up"
apt -y autoremove && apt -y autoclean

echo "installing pixi"
curl -fsSL https://pixi.sh/install.sh | bash 

echo "updating ~/.bashrc"
echo "alias bat=batcat" >> ~/.bashrc
echo "alias btop=btop --utf-force" >> ~/.bashrc
echo 'if [ -f ~/.bashrc ]; then source ~/.bashrc; fi' >> ~/.bash_profile
source ~/.bashrc

echo "creating project directories"
mkdir ~/dev ~/datasets ~/experiments ~/models

echo "cloning geovision and installing required packages"
cd ~/dev && git clone https://github.com/citruscaesar/geovision.git 
cd ~/dev/geovision
pixi install -e dev