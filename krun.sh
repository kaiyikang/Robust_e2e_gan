#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)



stage=0        # start from 0 if you need to start from data preparation
gpu=            # 已经不再使用，请使用ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # 使用多少 minibatches (主要是为了debug). "0" 代表使用所有的minibatch
verbose=0      # verbose option
resume=${resume:=none}    # Resume the training from snapshot


do_delta=false # 当使用CNN时，这里为true


etype=vggblstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
subsample_type="skip"

dlayers=1
dunits=300
atype=location
aact_func=softmax
aconv_chans=10
aconv_filts=100
lsm_type="none"
lsm_weight=0.0
dropout_rate=0.0

mtlalpha=0.5


batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=30

# rnnlm 相关
model_unit=char
batchsize_lm=64
dropout_lm=0.5
input_unit_lm=256
hidden_unit_lm=650
lm_weight=0.2
fusion=${fusion:=none}

# decoding parameter
lmtype=rnnlm
beam_size=12
nbest=12
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'



tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1; 

. ./cmd.sh
. ./path.sh 


if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi


set -e # 它使得脚本只要发生错误，就终止执行。
set -u # 脚本在头部加上它，遇到不存在的变量就会报错，并停止执行。
set -o pipefail # 只要一个子命令失败，整个管道命令就失败，脚本就会终止执行。


dataroot="/home/kang/Develop/Robust_e2e_gan/data/data_aishell"


train_set=train
train_dev=dev

recog_set="test"










# 创建一个checkpoint文件夹
lmexpdir=checkpoints/train_${lmtype}_2layer_${input_unit_lm}_${hidden_unit_lm}_drop${dropout_lm}_bs${batchsize_lm}
mkdir -p ${lmexpdir}



if [ ${stage} -le 3 ]; then

    echo "stage 3: LM Preparation"

    lmdatadir=${lmexpdir}/local/lm_train
    mkdir -p ${lmdatadir}

    text2token.py -s 1 -n 1 ${dataroot}/clean_text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' > ${lmdatadir}/train_trans.txt



    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt



fi
