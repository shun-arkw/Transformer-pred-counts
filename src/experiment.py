import argparse
import numpy as np
import yaml, os 
import torch 
from time import time 
import datetime
from zoneinfo import ZoneInfo

# from trainer import Trainer
from loader.data import _load_data, HybridDataCollator, DataCollatorForNumAdditions, GenerateLabels # , PolynomialDataCollator
from loader.model import load_model_for_num_additions
from trainer.utils import compute_metrics_for_num_additions, preprocess_logits_for_metrics, LimitStepsCallback
from misc.utils import count_cuda_devices
# np.seterr(all='raise')

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--data_dir",      type=str, default="./data/data_sum", help="Experiment dump path")
    parser.add_argument("--data_config_path",      type=str, default="./data/data_sum", help="Experiment dump path")
    parser.add_argument("--data_encoding",  type=str, default="infix")
    parser.add_argument("--save_path",      type=str, default="./dumped", help="Experiment dump path")
    parser.add_argument("--save_periodic",  type=int, default=0, help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_name",       type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id",         type=str, default="", help="Experiment ID")
    parser.add_argument("--group",          type=str, default="", help="Experiment group")
    parser.add_argument("--task",           type=str, default="sum", help="Task name")   
    parser.add_argument("--is_soft_labels",    type=bool, default=False, help="Use soft labels for classification") 
    parser.add_argument("--soft_label_temp",    type=float, default=1.0, help="Temperature for soft labels")

    # float16 / AMP API
    parser.add_argument("--fp16",           type=bool, default=True, help="Run model with float16")
    parser.add_argument("--amp",            type=int, default=2, help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--model",              type=str, default='bart', help="Embedding layer size")
    parser.add_argument("--d_model",            type=int, default=512, help="Embedding layer size")
    parser.add_argument("--dim_feedforward",    type=int, default=2048, help="feedforward layer size")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of Transformer layers in the encoder")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of Transformer layers in the decoder")
    parser.add_argument("--nhead",              type=int, default=8, help="Number of Transformer heads")
    parser.add_argument("--dropout",            type=float, default=0.1, help="Dropout")
    parser.add_argument("--attention_dropout",  type=float, default=0, help="Dropout in the attention layer")
    # parser.add_argument("--continuous_embedding_model", type=str, default='ffn')
    parser.add_argument("--encoding_method",    type=str, default='standard')
    parser.add_argument("--token_register_size", type=int, default=2)
    parser.add_argument("--positional_encoding", type=str, default='sinusoidal', choices=['sinusoidal', 'embedding'])
    
    # vocab and tokenizer parameters
    parser.add_argument("--num_variables", type=int, default=2)
    parser.add_argument("--field", type=str, default='QQ', help='QQ, RR, or GFP with some prime P (e.g., GF7).')
    parser.add_argument("--max_coefficient", type=int, default=1000, help='The maximum coefficients')
    parser.add_argument("--max_degree", type=int, default=10, help='The maximum degree')
    parser.add_argument("--weight_mx_entry_bound", type=int, default=1000, help='The maximum weight matrix entry bound')


    # training parameters
    parser.add_argument("--max_sequence_length", type=int, default=10000,
                        help="Maximum sequences length")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of sentences per batch")
    parser.add_argument("--test_batch_size", type=int, default=256,
                        help="Number of sentences per batch")
    parser.add_argument("--optimizer", type=str, default="adamw_apex_fused",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="learning rate (default 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight decay (default 0)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument("--max_steps_per_epoch", type=int, default=-1, help="Maximum epoch size")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of CPU workers for DataLoader")

    # environment parameters
    parser.add_argument("--resume_from_checkpoint", action='store_true', default=False)
    parser.add_argument("--dryrun", action='store_true', default=False) # --dryrunが与えられると，Trueになる
    
    ## Dev
    parser.add_argument('--regression_weight', type=float, default=0.1)
    # parser.add_argument("--continuous_coefficient", action='store_true', default=False)
    # parser.add_argument("--continuous_exponent", action='store_true', default=False)
    # parser.add_argument("--support_learning", action='store_true', default=False)
    

    return parser


def main():
    import wandb 
    from transformers import TrainingArguments
    from trainer.trainer import CustomTrainerForNumAdditons as Trainer
    from trainer.trainer import CustomTrainingArguments as TrainingArguments

    parser = get_parser()
    params = parser.parse_args()            

    os.makedirs(params.save_path, exist_ok=True)

    ## Load data
    with open(params.data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    train_data_path = os.path.join(params.data_dir, 'train.uni_F_matrix_nadds')
    test_data_path = os.path.join(params.data_dir, 'test.uni_F_matrix_nadds')
    trainset    = _load_data(train_data_path)
    testset     = _load_data(test_data_path)

    print(len(trainset), len(testset))
    

    if params.dryrun:
        trainset.input = trainset.input[:1000] # 1000
        trainset.target = trainset.target[:1000] # 1000
        params.epochs = 10
        params.save_path = os.path.join(os.path.dirname(params.save_path), 'dryrun')
        params.exp_name = 'dryrun'

    ## Load model
    ### standard embedding
    # if 'standard' in params.encoding_method:
    from dataset.tokernizer import set_tokenizer, set_vocab_for_num_additions
    
    use_continous_token = params.encoding_method == 'hybrid'
    
    vocab = set_vocab_for_num_additions(params.num_variables, 
                        field=params.field, 
                        max_coeff=params.max_coefficient, 
                        max_degree=params.max_degree, 
                        weight_mx_entry_bound=params.weight_mx_entry_bound, 
                        continuous_coefficient=use_continous_token, 
                        continuous_exponent=False)
    tokenizer = set_tokenizer(vocab)


    """ ラベル付与 """
    # bins = np.array([0, 100, 150, 200, 400, 1000, 2000, 3000, 4000, 5000])
    # bins = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    bins = np.array([0, 100, 200, 300, 400, 500])

    num_classes = len(bins) - 1
    temp = params.soft_label_temp
    print(f'num_classes: {num_classes}')
    print(f'Temperature for softmax labels: {temp}')

    generate_labels = GenerateLabels(bins, num_classes=num_classes, is_soft_labels=params.is_soft_labels, temp=temp)
    
    model = load_model_for_num_additions(params, num_classes=num_classes, vocab=vocab, tokenizer=tokenizer)
    
    if not use_continous_token:
        dc = DataCollatorForNumAdditions(tokenizer, generate_labels)
        label_names = ['labels']
    else:
        dc = HybridDataCollator(tokenizer)
        label_names = ['labels', 'labels_for_regression']

    tokenizer.save_pretrained(os.path.join(params.save_path, "tokenizer.json"))

        
    ## Save parameters 
    with open(os.path.join(params.save_path, "params.yaml"), "w") as f:
        yaml.dump(vars(params), f)

    now = datetime.datetime.now(ZoneInfo("Asia/Tokyo")) # Asia/Tokyoに変更
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    run_name = f'{params.exp_id}_{datetime_str}'

    ## Set up trainer
    trainer_config = TrainingArguments(
        output_dir                  = params.save_path,
        num_train_epochs            = params.epochs,
        # max_steps_per_epoch         = params.max_steps_per_epoch,
        logging_steps               = 50,
        save_total_limit            = 1,
        dataloader_pin_memory       = True,
        bf16                        = True,
        # save_steps                  = 100,
        eval_steps                  = 100,
        label_names                 = label_names, 
        remove_unused_columns       = False,
        per_device_train_batch_size = params.batch_size // count_cuda_devices(),
        eval_strategy               = 'steps',
        per_device_eval_batch_size  = params.batch_size // count_cuda_devices(), # つけたし
        # torch_compile               = True,
        report_to                   = "wandb",
        disable_tqdm                = True,
        run_name                    = run_name,
    )

    limit_steps_callback = LimitStepsCallback(max_steps_per_epoch=params.max_steps_per_epoch)

    

    # _compute_metrics = lambda x: compute_metrics(x, ignore_index=model.special_token_ids['pad_token_id'])
    _compute_metrics = lambda x: compute_metrics_for_num_additions(x)
    trainer = Trainer(
        args                            = trainer_config,
        model                           = model,
        train_dataset                   = trainset,
        eval_dataset                    = testset,
        data_collator                   = dc,
        compute_metrics                 = _compute_metrics,
        preprocess_logits_for_metrics   = preprocess_logits_for_metrics,
        # callbacks                       = [limit_steps_callback]
        )

    ## Run training
    wandb.init(project=params.exp_name, 
               name=run_name,
               group=params.group,
               config=trainer_config)

    s = time()
    train_result = trainer.train()
    print(f'training time: [{time()-s:.1f} sec]')

    trainer.save_model()
    
    metrics = train_result.metrics
    dataset_metrics = trainer.evaluate(metric_key_prefix="test")
    metrics.update(dataset_metrics)


    testloader = trainer.get_eval_dataloader()
    
    # scores = generation_accuracy_for_num_additions(model, num_classes, testloader, max_length=params.max_sequence_length, tokenizer=tokenizer, disable_tqdm=True)
    # metrics.update({'test generation accuracy': scores['acc']})
    trainer.save_metrics("all", metrics)
    wandb.log(metrics)

    wandb.finish()

if __name__ == '__main__':
    main()    
