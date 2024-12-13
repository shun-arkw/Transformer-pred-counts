task=pred_nadds
nvars=2
field=GF7
base_gb_type=shape

max_test_samples_for_input=10000
test_num_matrices_per_F=1

max_train_samples_for_input=100000
train_num_matrices_per_F=1

matrix_entry_distribution="uniform" # "uniform" , "normal", "grevlex"


data_name=${task}_n=${nvars}_field=${field} 
input_dir=data/${task}/${data_name}
output_dir=data/${task}/${data_name}/F_matrix_nadds

mkdir -p $output_dir
sage src/dataset/get_num_additions.sage  --input_dir $input_dir \
                                                --output_dir $output_dir \
                                                --buchberger_threshold 500 \
                                                --max_itrs 10 \
                                                --field $field \
                                                --nvars $nvars \
                                                --matrix_entry_lower_bound " -1000" \
                                                --matrix_entry_upper_bound 1000 \
                                                --matrix_entry_distribution $matrix_entry_distribution \
                                                --num_bins 10 \
                                                --upper_lim_for_num_tokens 1000 \
                                                --lower_lim_for_num_tokens 0 \
                                                --test_num_samples $max_test_samples_for_input \
                                                --train_num_samples $max_train_samples_for_input \
                                                --test_num_matrices_per_F $test_num_matrices_per_F \
                                                --train_num_matrices_per_F $train_num_matrices_per_F \
                                        # --testset_only 
                                        # --strictly_conditioned

