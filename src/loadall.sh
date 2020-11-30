#! /bin/sh
for d in ../data/fma_medium/*/; do
    # Will print */ if no directories are available
    SUBSTRING=$(echo $d | cut -c 20-22)
    if [ -f "../data/npy/${SUBSTRING}.npy" ]
    then
        echo "${SUBSTRING}.npy exists in ../data/npy... skipping"
    else
       python3 -W ignore load_audio.py $SUBSTRING
    fi
    
done
