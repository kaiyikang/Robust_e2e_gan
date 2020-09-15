#!/usr/bin/env bash

. ../path.sh
set -e
/home/kang/Develop/venv/bin/python3 prepare_feats_niubi.py /home/kang/Develop/Robust_e2e_gan/data/data_aishell /home/kang/Develop/Robust_e2e_gan/data/small_feats 1 
