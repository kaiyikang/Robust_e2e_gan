#!/usr/bin/env bash

# This script can unzip data_aishell and resource_aishell zip files.
# Some code come from kaldi/egs/aishell/s5/local/download_and_unter.sh

# 提前的准备
# 1.修改path.sh 中的 KALDI_ROOT，让它是你电脑里kaldi的地址
# 2.下载data_aishell 和 resource_aishell 到和这个script一个文件夹下
# 3.这个论文的作者，在根目录下已经有了一个utils文件夹，为了防止混淆，请将kaldi的utils文件夹软连接到这个目录

set -e
set -u
set -o pipefail 
cd `pwd`/data

stage=0


##################################################################

echo "====== Extract tgz ====== "
list="data_aishell.tgz resource_aishell.tgz"

if [ ${stage} -le 0 ];then

	for i in ${list};
	do   
	    # tar the files	    
	    if [ $i == data_aishell.tgz ];
	    then
		if [ ! -d data_aishell ];then
		    tar -xvzf $i
		
		    cd data_aishell/wav
		    for wav in ./*.tar.gz; do
		        echo "extracting wav from $wav"
		        tar -zxf $wav && rm $wav
		    done
		    
		    echo "Extracting data_aishell done."
		else
		    echo "$i has been extracted."
		fi
	    fi


	    if [ $i == resource_aishell.tgz ]
	    then
		tar -xvzf $i
		echo "Extracting resource_aishell done."
	    fi
	done
	echo "AISHELL extracting succeeded"
fi

################################################################

echo "===== Prepare dict ====="

if [ ${stage} -le 1 ];then

	res_dir=resource_aishell
	dict_dir=lang_syllable

	mkdir -p $dict_dir

	cp $res_dir/lexicon.txt $dict_dir

	cat $dict_dir/lexicon.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
	  perl -e 'while(<>){ chomp($_); $phone = $_; next if ($phone eq "sil");
	    m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$1} .= "$phone "; }
	    foreach $l (values %q) {print "$l\n";}
	  ' | sort -k1 > $dict_dir/nonsilence_phones.txt  || exit 1;

	echo sil > $dict_dir/silence_phones.txt

	echo sil > $dict_dir/optional_silence.txt


	cat $dict_dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dict_dir/extra_questions.txt || exit 1;
	cat $dict_dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
	  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
	 >> $dict_dir/extra_questions.txt || exit 1;

	echo "AISHELL dict preparation succeeded"
fi

################################################################

echo "===== Prepare data ====="

if [ ${stage} -le 2 ];then
	aishell_audio_dir=data_aishell/wav
	aishell_text=data_aishell/transcript/aishell_transcript_v0.8.txt

	train_dir=local/train
	dev_dir=local/dev
	test_dir=local/test
	tmp_dir=local/tmp

	mkdir -p $train_dir
	mkdir -p $dev_dir
	mkdir -p $test_dir
	mkdir -p $tmp_dir

	# data directory check
	if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
	  echo "Error: $0 requires two directory arguments"
	  exit 1;
	fi

	# find wav audio file for train, dev and test resp.
	find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
	n=`cat $tmp_dir/wav.flist | wc -l`
	[ $n -ne 141925 ] && \
	  echo Warning: expected 141925 data data files, found $n

	grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
	grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
	grep -i "wav/test" $tmp_dir/wav.flist > $test_dir/wav.flist || exit 1;

	rm -r $tmp_dir

	# Transcriptions preparation
	for dir in $train_dir $dev_dir $test_dir; do
	  echo Preparing $dir transcriptions
	  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
	  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
	  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
	  utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
	  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
	  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
	  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
	  sort -u $dir/transcripts.txt > $dir/text
	  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
	done

	mkdir -p clean_aishell/train clean_aishell/dev clean_aishell/test

	for f in spk2utt utt2spk wav.scp text; do
	  cp $train_dir/$f clean_aishell/train/$f || exit 1;
	  cp $dev_dir/$f clean_aishell/dev/$f || exit 1;
	  cp $test_dir/$f clean_aishell/test/$f || exit 1;
	done
	echo "AISHELL data preparation succeeded"
fi

################################################################

echo "===== Prepare Lang ====="

if [ ${stage} -le 3 ];then
	utils/prepare_lang.sh --position-dependent-phones false lang_syllable \
	    "<SPOKEN_NOISE>" local/lang clean_aishell/lang || exit 1;
# srcdir=$1 oov_word=$2 tmpdir=$3 dir=$4
# <SPOKEN_NOISE>
fi

echo "AISHELL Lang preparation succeed"

echo "All done"

cd `pwd`/..




