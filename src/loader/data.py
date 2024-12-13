import os 
import torch 
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List
import numpy as np 
import random


def _load_data(data_path):
    try:
        with open(data_path, "r") as f:
            data = f.read().splitlines()
    except:
        raise FileNotFoundError
    
    input_texts = [line.split(":")[0].strip() for line in data] # strip()で両端の空白文字列を削除
    target_texts = [line.split(":")[1].strip() for line in data]


    input_texts_shuffled = random.sample(input_texts, len(input_texts))
    target_texts_shuffled = random.sample(target_texts, len(target_texts))

    dataset = DictDataset(input_texts_shuffled, target_texts_shuffled)
    
    return dataset

def load_data(data_path, 
            encoding='prefix', 
            batch_sizes=[4, 100], 
            return_dataloader=True, 
            extensions=['train', 'test'], 
            do_shuffle=[True, False], 
            tokenizer=None,
            continuous_coefficient=True,
            continuous_exponent=False,
            support_learning=False,):
    
    
    ret = []
    for ext, batch_size, shuffle in zip(extensions, batch_sizes, do_shuffle):
        path = f"{data_path}.{ext}"
        print(f'loading ... {path}')
        if encoding: path = path + f'.{encoding}'
        dataset = _load_data(path)

        if return_dataloader: 
            data_collator = DataCollator(tokenizer, continuous_coefficient=continuous_coefficient, continuous_exponent=continuous_exponent, support_learning=support_learning)
            print(f'content of batch_size: {batch_size}', flush=True)
            dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True, collate_fn=data_collator)

        ret.append(dataset)

    return ret[0] if len(ret) == 1 else ret

class DictDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer=None):
        self.tokenizer = tokenizer

        "----------------tokenizer=Noneときはトークン化されない----------------"
        input_ = input_texts if tokenizer is None else tokenizer(input_texts, padding='longest', return_tensors='pt')
        target = target_texts if tokenizer is None else tokenizer(target_texts, padding='longest', return_tensors='pt')
        
        self.input = input_ if tokenizer is None else input_['input_ids']
        self.input_mask = None if tokenizer is None else input_['attention_mask'].bool()
        self.target = target if tokenizer is None else target['input_ids']
        self.target_mask = None if tokenizer is None else target['attention_mask'].bool()
        
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "input_mask": self.input_mask[idx] if self.tokenizer is not None else None,
            "target_mask": self.target_mask[idx] if self.tokenizer is not None else None,
        }

def str_to_float(s):
    try:
        return float(s)
    except:
        if '/' in s: 
            a, b = s.split('/')
            return float(a) / float(b)

        raise ValueError(f'invalid string: {s}')
    
class SimpleDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def __call__(self, batch):
        eos = self.tokenizer.eos_token
        input_texts = [item["input"] + f' {eos}' for item in batch]
        target_texts = [item["target"] + f' {eos}' for item in batch]
        
        input_encodings = self.tokenizer(input_texts, padding='longest', return_tensors='pt') # input_textsをトークン化して，パディングを最長のシーケンスに合わせて行う
        target_encodings = self.tokenizer(target_texts, padding='longest', return_tensors='pt')

        return {
            'encoder_input': input_encodings['input_ids'] ,
            'decoder_input': target_encodings['input_ids'],
            'encoder_padding_mask': ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'decoder_padding_mask': ~target_encodings['attention_mask'].bool(),
            'labels': target_encodings['input_ids'].contiguous(),
        }
    
# データコレータ内部で多項式加算数がどのレンジにあるかを判定してあげるようにする
# 多項式加算数を予測するタスク用のデータコレータ
class DataCollatorForNumAdditions:
    def __init__(self, tokenizer, generate_labels=None):
        self.tokenizer = tokenizer
        self.generate_labels = generate_labels

    @torch.no_grad()
    def __call__(self, batch):
        eos = self.tokenizer.eos_token # </s>
        input_texts = [item["input"] + f' {eos}' for item in batch]
        
        num_additions_list = [int(item["target"]) for item in batch]
        
        input_encodings = self.tokenizer(input_texts, padding='longest', return_tensors='pt') # input_textsをトークン化して，パディングを最長のシーケンスに合わせて行う


        if self.generate_labels is None:
            raise ValueError("generate_labels is None. Please provide a valid generate_labels function.")
        
        labels = self.generate_labels(num_additions_list) 

        return {
            'encoder_input': input_encodings['input_ids'],
            'encoder_padding_mask': ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'labels': labels,
        }


def _preprocess_coefficients(input_text: str):
    
    tokens = input_text.split()
    c_labels = [str_to_float(t[1:]) if t[0] == 'C' else np.nan for t in tokens]
    
    for i, _ in enumerate(tokens):
        if c_labels[i] is not np.nan:
            tokens[i] = '[C]'

    text = ' '.join(tokens)
    
    return (text, c_labels)
    
def preprocess_coefficients(input_texts: List[str]):
    return [_preprocess_coefficients(it) for it in input_texts]


class HybridDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        target_texts = [item["target"] for item in batch]

        eos = self.tokenizer.eos_token
        input_texts = [item["input"] + f' {eos}' for item in batch]
        target_texts = [item["target"] + f' {eos}' for item in batch]
        
        input_texts, input_coeff_labels = list(zip(*preprocess_coefficients(input_texts)))
        target_texts, target_coeff_labels = list(zip(*preprocess_coefficients(target_texts)))
        
        input_encodings     = self.tokenizer(input_texts, padding='longest', return_tensors='pt')
        target_encodings    = self.tokenizer(target_texts, padding='longest', return_tensors='pt')
        
        input_ids = input_encodings['input_ids'] 
        target_ids = target_encodings['input_ids']             
        
        length_in, length_tar = input_ids.shape[-1], target_ids.shape[-1]
        
        input_continuous_labels  = torch.tensor([t + [np.nan]*(length_in - len(t) ) for t in input_coeff_labels]).contiguous()
        target_continuous_labels = torch.tensor([t + [np.nan]*(length_tar - len(t)) for t in target_coeff_labels]).contiguous()
        # input_continuous_labels.append(input_coeff_labels.unsqueeze(-1))
        # target_continuous_labels.append(target_coeff_labels.unsqueeze(-1))
        
        return {
            'encoder_input': input_encodings['input_ids'] ,
            'decoder_input': target_encodings['input_ids'],
            'encoder_padding_mask': ~input_encodings['attention_mask'].bool(),  # NOTE: attantion mask given by tokenizer is 0/1 multiplicative mask (0 for no attention) but transformer use bool mask (True for no attention)
            'decoder_padding_mask': ~target_encodings['attention_mask'].bool(),
            'labels': target_encodings['input_ids'].contiguous(),
            'labels_for_regression': target_continuous_labels,
            'encoder_input_labels': input_continuous_labels,
            'decoder_input_labels': target_continuous_labels,
        }

        return {
            "input_ids"                 : input_ids,
            "attention_mask"            : attention_mask,
            "decoder_input_ids"         : target_ids[:, :-1].contiguous(),
            "decoder_attention_mask"    : target_attention_mask,
            "labels"                    : labels.contiguous(),
            "continuous_labels"         : continuous_labels,
            "input_continuous_labels"   : input_continuous_labels,
            "target_continuous_labels"  : target_continuous_labels,
        }

class GenerateLabels():
    def __init__(self, bins, num_classes, is_soft_labels=False, temp=1.0):

        # NumPyの配列であるかを確認
        if not isinstance(bins, np.ndarray):
            raise TypeError("エラー: binsはNumPyの配列である必要があります。")
    
        # 次元が1であるかを確認
        if bins.ndim != 1:
            raise ValueError("エラー: binsは一次元の配列である必要があります。")
        
        self.bins = bins # Array of bins. It has to be 1-dimensional and monotonic. e.g. np.array([0, 100, 200, 300, 400, 500])
        self.is_soft_labels = is_soft_labels
        self.num_classes = num_classes
        self.temp = temp

    def __call__(self, num_additions_list):
        num_additions_list = np.array(num_additions_list)

        # 多項式加算数がどのレンジにあるかを判定する
        labels = np.digitize(num_additions_list, self.bins) - 1 # 最初のbinに含まれるデータのラベルは0


        if self.is_soft_labels: # ソフトラベルを取得する
            soft_labels = np.zeros((0, self.num_classes))

            for label in labels:
                soft_labels = np.vstack((soft_labels, self.get_soft_label(label)))

            return torch.tensor(soft_labels).float() # 二次元テンソル(データ数 * num_classes)
        
        else: # ハードラベルを取得する
            hard_labels = np.eye(self.num_classes)[labels] # one-hotベクトルに変換
            return torch.tensor(hard_labels, dtype=torch.long) # 一次元テンソル(データ数)

    def get_soft_label(self, correct_class_id):
        temp = self.temp
        assert 0 <= correct_class_id <= self.num_classes - 1, '0 <= correct_class_id <= num_classes - 1 である必要がある'
        
        # ソフトマックス関数の処理
        diffs = np.ones(self.num_classes) * correct_class_id - np.arange(self.num_classes)
        dist = np.abs(diffs)
        get_soft_label = np.exp(-dist / temp) / np.sum(np.exp(-dist / temp))

        return get_soft_label
