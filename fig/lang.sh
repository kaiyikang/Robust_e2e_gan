#!/usr/bin/env bash
. ./path.sh
utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SPOKEN_NOISE>" data/local/lang data/lang 
