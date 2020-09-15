#!/usr/bin/env bash

# This script can unzip data_aishell and resource_aishell zip files.
# Some code come from kaldi/egs/aishell/s5/local/download_and_unter.sh

# This script should be located with two downloaded tgz files.

list="data_aishell.tgz resource_aishell.tgz"

for i in $list;
do
    if [ ! -e $i ];then
        echo "File $i does not exist."
        continue
    fi
    # tar the files
    echo "Load file :$i"
    if ! tar -xvzf $i; then
		echo "Error un-tarring archive $i"
		exit 1;
	else
		echo "Extracting $i successful."
        echo "Start to continue ..."
	fi
    
    if [ $i == data_aishell.tgz ];
    then
        cd data_aishell/wav
        for wav in ./*.tar.gz; do
            echo "Extracting wav from $wav"
            tar -zxf $wav && rm $wav
        done
        echo "Extracting data_aishell done."
    fi

    if [ $i == resource_aishell.tgz ]
    then
        echo "Extracting resource_aishell done."
    fi
done

