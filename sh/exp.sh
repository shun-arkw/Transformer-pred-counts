wandb_project_name=neurips-cr
gpu_id=0

# task=shape
task=pred_nadds

nvars=3
field=GF7
max_coefficient=200  # should be higher than coeff_bound
max_degree=40  # should be higher than max_degree_F * 2 + max_degree_G



encoding_method=standard
# encoding_method=hybrid

positional_encoding=embedding
# positional_encoding=sinusoidal

# epochs=8

epochs=10
# batch_size=16
batch_size=128

data_name=${task}_n=${nvars}_field=${field} #_init
data_path=data/${task}/${data_name}/data_${field}_n=${nvars}
data_config_path=config/${data_name}.yaml
data_dir=data/${task}/${data_name}/uniform_dataset

group=${encoding_method}_${positional_encoding} #_init
_save_path=${field}_n=${nvars}_ep=${epochs}_bs=${batch_size}
save_path=results/${task}/${group}/${_save_path}
run_name=${task}_${_save_path}

mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$gpu_id python3 src/experiment.py  --save_path $save_path \
                                            --data_dir $data_dir \
                                            --data_config_path $data_config_path \
                                            --task $task \
                                            --num_variables $nvars \
                                            --field $field \
                                            --max_coefficient $max_coefficient \
                                            --max_degree $max_degree \
                                            --epochs $epochs \
                                            --batch_size $batch_size \
                                            --test_batch_size $batch_size \
                                            --group $group \
                                            --exp_name $wandb_project_name \
                                            --exp_id $run_name \
                                            --is_soft_labels True \
                                            --soft_label_temp 2.0 \
                                            --positional_encoding $positional_encoding \
                                            --encoding_method $encoding_method # > ${save_path}/run.log &
                                            # --dryrun \

                                            # --max_steps_per_epoch $max_steps_per_epoch \
                                            # --positional_encoding embedding \

gpu_id=$((gpu_id+1))

# done