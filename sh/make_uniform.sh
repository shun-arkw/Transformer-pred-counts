task=pred_nadds
nvars=3
field=GF7
base_gb_type=shape

train_num_samples=1200
test_num_samples=130



data_name=${task}_n=${nvars}_field=${field} 
input_dir=data/${task}/${data_name}/F_matrix_nadds
output_dir=data/${task}/${data_name}/uniform_dataset

mkdir -p $output_dir
python3 src/dataset/make_uniform.py  --input_dir $input_dir \
                                     --output_dir $output_dir \
                                     --train_num_samples $train_num_samples \
                                     --test_num_samples $test_num_samples \
                                     --field $field \
                                     --nvars $nvars \
