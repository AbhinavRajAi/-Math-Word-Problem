#!/bin/bash




var_update_modes=(gru)
bert_model_names=(roberta-base)
cuda_devices=(0)

for (( d=0; d<${#var_update_modes[@]}; d++  )) do
    var_update_mode=${var_update_modes[$d]}
    for (( e=0; e<${#bert_model_names[@]}; e++  )) do
        bert_model_name=${bert_model_names[$e]}
        dev=${cuda_devices[$e]}
        model_folder=svamp_${bert_model_name}_${var_update_mode}
        echo "Running svamp with bert model $bert_model_name and var update mode $var_update_mode"
        TOKENIZERS_PARALLELISM=false \
        python3 /home/abhiraj/DeductiveReasoning/Deductive-MWP/universal_main1.py --device=cuda:${dev} --model_folder=${model_folder} --mode=train --height=7 --num_epochs=1000 \
                          --train_file=/home/abhiraj/DeductiveReasoning/Deductive-MWP/data/mawps_asdiv-a_svamp/modified_equation_train.json  --train_num=-1 --dev_num=-1 --var_update_mode=${var_update_mode} \
                          --dev_file=/home/abhiraj/DeductiveReasoning/Deductive-MWP/data/mawps_asdiv-a_svamp/modified_equation_test.json --test_file=none --bert_model_name=${bert_model_name}  --fp16=1 \
                           --learning_rate=2e-5 > /home/abhiraj/DeductiveReasoning/Deductive-MWP/logs/${model_folder}.log 2>&1
    done
done




