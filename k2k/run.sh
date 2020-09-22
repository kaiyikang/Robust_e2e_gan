#!/usr/bin/env bash

cd `pwd`

pyvenv=/home/kang/Develop/venv/bin/python3


# general configuration
stage=4        # start from 0 if you need to start from data preparation
gpu=           # 已经不再使用，请使用ngpu
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # 使用多少 minibatches (主要是为了debug). "0" 代表使用所有的minibatch
verbose=0      # verbose option
resume=${resume:=none}    # Resume the training from snapshot

# feature configuration
do_delta=false # 当使用CNN时，这里为true

# 网络结构
# 编码器encoder 相关的
#etype=vggblstmp     # encoder architecture type
etype=blstmp
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
subsample_type="skip"
# 解码器相关decoder
dlayers=1
dunits=300
# 注意力attention相关
atype=location
aact_func=softmax
aconv_chans=10
aconv_filts=100
lsm_type="none"
lsm_weight=0.0
dropout_rate=0.0

# hybrid CTC/attention 结合了ctc attention的参数
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=30

# rnnlm
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


# exp tag
tag="" # tag for managing experiments.

##################### for bash 
. utils/parse_options.sh || exit 1; # 在sh中使用parameters

. ./cmd.sh # 配置 运行和训练的 memory和 gpu

. ./path.sh # 配置 路径

# 检查gpu使用的设置 ，因为$gpu 已经不再使用 check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi


set -e          # 它使得脚本只要发生错误，就终止执行。
set -u          # 脚本在头部加上它，遇到不存在的变量就会报错，并停止执行。
set -o pipefail # 只要一个子命令失败，整个管道命令就失败，脚本就会终止执行。

# 需要更改成对应的路径
dataroot="/home/kang/Develop/Robust_e2e_gan/k2k/data/clean_aishell"
##dictroot="/home/bliu/mywork/workspace/e2e/data/lang_1char/"
dictroot="/home/kang/Develop/Robust_e2e_gan/k2k/data/lang_syllable"
train_set=train
train_dev=dev
##recog_set="test_mix test_clean"
##recog_set="test_clean_small"
recog_set="test"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}_units.txt

embed_init_file=${dictroot}/char_embed_vec

echo "dictionary: ${dict}"

nlsyms=${dictroot}/non_lang_syms.txt # list of non-linguistic symobles, e.g., <NOISE> etc.


lmexpdir=checkpoints/train_${lmtype}_2layer_${input_unit_lm}_${hidden_unit_lm}_drop${dropout_lm}_bs${batchsize_lm}
mkdir -p ${lmexpdir}


if [ ${stage} -le 1 ]; then
    echo "stage 1: data and dict Preparation"
    . ./data/data_prep.sh || exit 1;
    cp data/clean_aishell/train/text data/clean_aishell/train/text_syllable
    cp data/clean_aishell/dev/text data/clean_aishell/dev/text_syllable
    echo "done"
fi


if [ ${stage} -le -1 ];then
    echo "stage 2 : mix clean and noise"
    echo `pwd`
    ${pyvenv} ./data/prepare_feats.py data  data/feats 1
fi


if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"


    lmdatadir=${lmexpdir}/local/lm_train
    mkdir -p ${lmdatadir}

    text2token.py -s 1 -n 1  ${dataroot}/train/text_syllable | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
        # 删除了 -l ${nlsyms}
        
    ##text2token.py -s 1 -n 1 -l ${nlsyms} /home/bliu/mywork/workspace/e2e/data/lang_1char/all_text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
    ##    > ${lmdatadir}/train_trans.txt
    

    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1  ${dataroot}/${train_dev}/text_syllable | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt


    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi

    echo "text2token done"
    
    ${pyvenv} utils/build_vocab.py 'data/clean_aishell/train/text' 3 ${dict}

    
    # 这里使用的是venv 中的python
    ${cuda_cmd} ${lmexpdir}/train.log ${pyvenv} lm_train.py --ngpu 1 --input-unit ${input_unit_lm} --lm-type ${lmtype} --unit ${hidden_unit_lm} --dropout-rate ${dropout_lm} --verbose 1 --batchsize ${batchsize_lm} --outdir ${lmexpdir} --train-label ${lmdatadir}/train.txt  --valid-label ${lmdatadir}/valid.txt --dict ${dict} --embed-init-file ${dictroot}/sgns-wiki

fi

name=aishell_${model_unit}_${etype}_e${elayers}_subsample${subsample}_${subsample_type}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_${aact_func}_aconvc${aconv_chans}_aconvf${aconv_filts}_lsm_type${lsm_type}_lsm_weight${lsm_weight}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_dropout${dropout_rate}_fusion${fusion}
##name=aishell_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}  --resume $resume \
lmexpdir=checkpoints/train_fsrnnlm_2layer_256_650_drop0.5_bs64

if [ ${stage} -le 4 ]; then
    echo "stage 4: E2E Network Training"

    ${pyvenv} asr_train_fbank.py \
    --dataroot $dataroot \
    --name $name \
    --model-unit $model_unit \
    --resume $resume \
    --dropout-rate ${dropout_rate} \
    --etype ${etype} \
    --elayers ${elayers} \
    --eunits ${eunits} \
    --eprojs ${eprojs} \
    --subsample ${subsample} \
    --subsample-type ${subsample_type} \
    --dlayers ${dlayers} \
    --dunits ${dunits} \
    --atype ${atype} \
    --aact-fuc ${aact_func} \
    --aconv-chans ${aconv_chans} \
    --aconv-filts ${aconv_filts} \
    --mtlalpha ${mtlalpha} \
    --batch-size ${batchsize} \
    --maxlen-in ${maxlen_in} \
    --maxlen-out ${maxlen_out} \
    --opt_type ${opt} \
    --verbose ${verbose} \
    --lmtype ${lmtype} \
    --rnnlm ${lmexpdir}/rnnlm.model.best \
    --fusion ${fusion} \
    --epochs ${epochs} \
    --feat_type 'fft' \
    --fbank_dim 80
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Enhance base  Training"

    ${pyvenv} enhance_base_train.py \
    --dataroot $dataroot 
    
fi
