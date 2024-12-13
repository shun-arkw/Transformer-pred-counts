task=pred_nadds #shape, cauchy, pred_nadds
num_variables=3
field=GF7
# field=QQ

# shapeからnon-GBを生成
# yamlでデータセットのサンプル数は指定できる

config=${task}_n=${num_variables}_field=${field} #_init
save_dir=data/${task}/${config}

mkdir -p $save_dir
sage src/dataset/build_dataset.sage     --save_path $save_dir \
                                        --strictly_conditioned \
                                        --config_path config/${config}.yaml # > ${save_dir}/run_${config}.log
                                        # --testset_only 
                                        # --strictly_conditioned
