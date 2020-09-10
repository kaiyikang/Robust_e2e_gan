#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
stage=6        # start from 0 if you need to start from data preparation
gpu=            # 已经不再使用，请使用ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # 使用多少 minibatches (主要是为了debug). "0" 代表使用所有的minibatch
verbose=0      # verbose option
resume=${resume:=none}    # Resume the training from snapshot

# feature configuration
do_delta=false # 当使用CNN时，这里为true

# 网络结构
# 编码器encoder 相关的
etype=vggblstmp     # encoder architecture type
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


# exp tag
tag="" # tag for managing experiments.

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

# set命令用来修改 Shell 环境的运行参数，也就是可以定制环境。
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e # 它使得脚本只要发生错误，就终止执行。
set -u # 脚本在头部加上它，遇到不存在的变量就会报错，并停止执行。
set -o pipefail # 只要一个子命令失败，整个管道命令就失败，脚本就会终止执行。

# 需要更改成对应的路径
dataroot="/home/bliu/SRC/workspace/e2e/data/clean_aishell/"
##dictroot="/home/bliu/mywork/workspace/e2e/data/lang_1char/"
dictroot="/home/bliu/SRC/workspace/e2e/data/lang_syllable/"
train_set=train
train_dev=dev
##recog_set="test_mix test_clean"
##recog_set="test_clean_small"
recog_set="test"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}_units.txt

embed_init_file=${dictroot}/char_embed_vec

echo "dictionary: ${dict}"

nlsyms=${dictroot}/non_lang_syms.txt

# 创建一个checkpoint文件夹
lmexpdir=checkpoints/train_${lmtype}_2layer_${input_unit_lm}_${hidden_unit_lm}_drop${dropout_lm}_bs${batchsize_lm}
mkdir -p ${lmexpdir}

# 开始执行step3，训练语言网络
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    # 创建checkpoint
    lmdatadir=${lmexpdir}/local/lm_train
    mkdir -p ${lmdatadir}
    # 以text_syllable 为输入作为 text2token的参数
    # py 文件可以直接运行，原因是在py文件中加入了头部注释
    # -s 是skip first n columns，忽略第一列
    # -n 是以几个字符为准，进行分割，例如 aabb -n 1 就是 a a b b
    # -l list of non-liguistic symbols 
    # 最后是input text

    # cut -f 显示指定字段的内容 -d 制定字段的分割符，例子：https://man.linuxde.net/cut
    # perl -e 执行代码 > 并且输出train_trans.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dataroot}/train/text_syllable | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    ##text2token.py -s 1 -n 1 -l ${nlsyms} /home/bliu/mywork/workspace/e2e/data/lang_1char/all_text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
    ##    > ${lmdatadir}/train_trans.txt
    
    # 将train_trans.txt 中的 \n 替换 'space' 并且 train.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    # 同样处理一下 ${train_dev} 中的 > valid.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} ${dataroot}/${train_dev}/text_syllable | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt

    # use only 1 gpu
    # 如果 ${ngpu} > 1 
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi

    # 调用 cuda_cmd 输出 train.log 日志文件
    # 运行 lm_train.py 训练语言模型
    ${cuda_cmd} ${lmexpdir}/train.log \
        python3 lm_train.py \
        --ngpu 1 \ # 使用gpu的数量，默认为0, 这里设定为1
        --input-unit ${input_unit_lm} \ # Number of LSTM units in each layer
        --lm-type ${lmtype} \ # 可以选择:rnnlm fsrnnlm
        --unit ${hidden_unit_lm} \ # Number of LSTM units in each layer
        --dropout-rate ${dropout_lm} \ 
        --embed-init-file ${embed_init_file} \ # Dictionary
        --verbose 1 \ # 设定为1，使用logging输出日志
        --batchsize ${batchsize_lm} \ # Number of examples in each mini-batch
        --outdir ${lmexpdir} \ # 输出路径
        --train-label ${lmdatadir}/train.txt \ # Filename of train label data (json)
        --valid-label ${lmdatadir}/valid.txt \ # Filename of validation label data (json)
        --dict ${dict} # Dictionary
fi

# 创建checkpoint文件夹
name=aishell_${model_unit}_${etype}_e${elayers}_subsample${subsample}_${subsample_type}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_${aact_func}_aconvc${aconv_chans}_aconvf${aconv_filts}_lsm_type${lsm_type}_lsm_weight${lsm_weight}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_dropout${dropout_rate}_fusion${fusion}
##name=aishell_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}  --resume $resume \
lmexpdir=checkpoints/train_fsrnnlm_2layer_256_650_drop0.5_bs64

# 开始step4，训练神经网络
if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    # 调用asr_train.py 训练神经网络
    python3 asr_train.py \
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
    --epochs ${epochs}
fi

expdir=checkpoints/asr_clean_syllable_fbank80_drop0.2/
name=asr_clean_syllable_fbank80_drop0.2
lmexpdir=checkpoints/train_rnnlm_2layer_256_650_drop0.2_bs64
fst_path="/home/bliu/mywork/workspace/e2e/data/lang_word/LG_pushed_withsyms.fst"
nn_char_map_file="/home/bliu/mywork/workspace/e2e/data/lang_word/net_chars.txt"

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=4
    for rtask in ${recog_set}; do
    ##( 
    	echo "stage 5: 检查，rtask变量：${rtask}"
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_${lmtype}${lm_weight}
        feat_recog_dir=${dataroot}/${rtask}/
        # run fix data dir 
	# 这个脚本确保只有 "feats.scp"、"wav.scp"[如果存在的话]、segments[如果存在的话]text和utt2spk中的片段存在。
	utils/fix_data_dir.sh $feat_recog_dir
        # split data 
        ##splitjson.py --parts ${nj} ${feat_recog_dir}/data.json  --kenlm ${dictroot}/text.arpa \
        sdata=${feat_recog_dir}/split$nj
        mkdir -p ${expdir}/${decode_dir}/log/
	# 切割数据
	# [[]] 条件判断 
	# -d  如果 file 存在并且是一个目录，则为true。
	# [ file1 -ot file2 ]：如果 FILE1 比 FILE2 的更新时间更旧，或者 FILE2 存在而 FILE1 不存在，则为true
        [[ -d $sdata && ${feat_recog_dir}/feats.scp -ot $sdata ]] || utils/split_data.sh ${feat_recog_dir} $nj || exit 1;
        echo $nj > ${expdir}/num_jobs

        #### use CPU for decoding  ##& ##${decode_cmd} JOB=1 ${expdir}/${decode_dir}/log/decode.JOB.log \

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            python3 asr_recog.py \
            --dataroot ${dataroot} \
            --name $name \
            --model-unit $model_unit \
            --nj $nj \
            --gpu_ids 0 \
            --nbest $nbest \
            --resume ${expdir}/model.acc.best \
            --recog-dir ${sdata}/JOB/ \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lmtype ${lmtype} \
            --verbose ${verbose} \
            --normalize_type 0 \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --fstlm-path ${fst_path} \
            --nn-char-map-file ${nn_char_map_file} \
            --lm-weight ${lm_weight} 
        # 不是由kaldi提供的脚本
        score_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        
        ##kenlm_path="/home/bliu/mywork/workspace/e2e/src/kenlm/build/text_character.arpa"
        ##rescore_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${expdir}/${decode_dir}_rescore ${dict} ${kenlm_path}
    ##) &
    done
    ##wait
    echo "Finished"
fi
