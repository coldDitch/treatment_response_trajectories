#!/bin/bash
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
install_cmdstan
mkdir src/logs
mkdir src/posterior_data