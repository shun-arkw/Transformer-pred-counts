import os
import time
import argparse
import numpy as np
import yaml
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


load('src/dataset/symbolic_utils.sage')
load('src/dataset/groebner.sage')
load('src/dataset/count_num_additions.sage')

np.random.seed((os.getpid() * int(time())) % 123456789)



def output_static(data_nadds, threshold, num_bins=10):
    nadds_static = {}
    num_samples = len(data_nadds)
    nadds_static['mean'] = float(np.mean(data_nadds))
    nadds_static['std'] = float(np.std(data_nadds))
    nadds_static['max'] = float(np.max(data_nadds))
    nadds_static['min'] = float(np.min(data_nadds))
    nadds_static['median'] = float(np.median(data_nadds))
    nadds_static['num_samples'] = int(num_samples)

    ret = plt.hist(data_nadds, bins=num_bins, range=(int(0), threshold), density=False)  
    counts, bin_edges = ret[0], ret[1]
    counts = counts.astype(int)

    for i in range(num_bins):
        nadds_static[f'{bin_edges[i]} <= num_additions < {bin_edges[i+1]}'] = int(counts[i])

    nadds_static[f'{bin_edges[num_bins]} <= num_additions'] = int(num_samples - np.sum(counts))

    return nadds_static


def load_data(input_dir, filename, num_samples):
    input_path = os.path.join(input_dir, filename)
    with open(input_path, 'r') as file:
        contents = file.read()

    F_G_str_list = contents.split('\n') # 多項式系を要素として持つリスト

    return F_G_str_list[:num_samples]

    
class BuildDataset():
    def __init__(self, tag, F_G_str_list, nvars, field, max_itrs, threshold, matrix_entry_lower_bound, matrix_entry_upper_bound, matrix_entry_distribution, lower_lim_for_num_tokens, upper_lim_for_num_tokens, num_matrices_per_F, n_jobs=-1, ):
        
        self.tag = tag
        self.F_G_str_list = F_G_str_list
        self.nvars = nvars
        self.field = field
        self.max_itrs = max_itrs
        self.threshold = threshold
        self.matrix_entry_lower_bound = matrix_entry_lower_bound
        self.matrix_entry_upper_bound = matrix_entry_upper_bound
        self.matrix_entry_distribution = matrix_entry_distribution
        self.lower_lim_for_num_tokens = lower_lim_for_num_tokens
        self.upper_lim_for_num_tokens = upper_lim_for_num_tokens
        self.num_matricies_per_F = num_matrices_per_F
        self.n_jobs = n_jobs

        
    def __call__(self):

        if self.tag == 'train':
            plus = 10000
        else:
            plus = 0

        ret = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(delayed(self.repeat)(F_G_str, seed=i+plus) for i, F_G_str in enumerate(self.F_G_str_list))
 

        data_str_list = [data_str for pair in ret for data_str, _ in pair if data_str is not None]
        data_num_additions_list = [num_additions for pair in ret for _, num_additions in pair if num_additions is not None]

        return data_str_list, data_num_additions_list
    
    def repeat(self, F_G_str, seed):
        np.random.seed(seed)
        n = self.num_matricies_per_F

        if self.matrix_entry_distribution == 'grevlex':
            n = 1

        pair_list = []
        for _ in range(n):
            pair = self.get_matrix_nadds(F_G_str)
            pair_list.append(pair)

        return pair_list


    def get_matrix_nadds(self, F_G_str):

        F, F_str, num_tokens_F = self.preprocess_for_F(F_G_str)

        upper = self.upper_lim_for_num_tokens
        lower = self.lower_lim_for_num_tokens

        # Fのトークン数が指定範囲外の場合は無視
        if upper is not None and lower is not None:
            if num_tokens_F > upper or num_tokens_F < lower:
                return None, None
                
            
        for i in range(self.max_itrs):

            weight_matrix = self.generate_weight_matrix()

            num_additions_counter = NumAdditionsCounter(weight_matrix,
                                                        polynomial_list = F,
                                                        coeff_field = self.field,
                                                        num_variables = self.nvars,
                                                        select_strategy = 'normal',
                                                        stop_algorithm = True,
                                                        threshold = self.threshold)
            _, nadds = num_additions_counter.run()

            if nadds <= self.threshold or i == self.max_itrs - 1:
                weight_matrix = weight_matrix.reshape(-1)
                nadds_str = str(nadds)
                break
        
        weight_matrix_str = ''
        for elem in weight_matrix:
            weight_matrix_str += 'W' + str(elem) + ' '
        weight_matrix_str = weight_matrix_str.strip()

        data_str = '[CLS] ' + weight_matrix_str + ' [SUPSEP] ' + F_str + ' : ' + nadds_str

        return data_str, nadds

    def preprocess_for_F(self, F_G_str):
        F_str = F_G_str.split(':')[0]
        F_str = F_str.strip()
        num_tokens_F = int(len(F_str.split()))
        F_list = F_str.split('[SEP]')

        Ring = PolynomialRing(self.field, 'x', self.nvars)
        F = [sequence_to_poly(f_str.strip(), Ring) for f_str in F_list] # infix以外はうまくいかない
        
        return F, F_str, num_tokens_F
    
    # 有効なweight matrixを生成する
    def generate_weight_matrix(self):
        distribution = self.matrix_entry_distribution
        while True:
            if distribution == 'uniform':
                weight_matrix = np.random.randint(self.matrix_entry_lower_bound, self.matrix_entry_upper_bound, size=(self.nvars, self.nvars))
                weight_matrix[0, :] = np.abs(weight_matrix[0, :])

            elif distribution == 'normal':
                weight_matrix = np.random.normal(0, 10, size=(self.nvars, self.nvars))
                weight_matrix = weight_matrix.astype(int)
                weight_matrix[0, :] = np.abs(weight_matrix[0, :])

            elif distribution == 'grevlex':
                first_row = np.ones((1, self.nvars), dtype=int)
                rows = np.eye(self.nvars, dtype=int)
                rows = np.flip(np.eye(self.nvars, dtype=int), axis=1)
                rows = -rows
                weight_matrix = np.concatenate([first_row, rows])
                weight_matrix = weight_matrix[:self.nvars, :]

            else:
                raise ValueError('Invalid matrix_entry_distribution')

            if validation(weight_matrix):
                break

        return weight_matrix


class Writer():
    def __init__(self, data_str_list, data_num_additions_list, output_dir, tag):

        self.data_str_list = data_str_list
        self.data_num_additions = np.array(data_num_additions_list)
        self.output_dir = output_dir
        self.tag = tag

    def write(self, filename):
        data_str = '\n'.join(self.data_str_list)
        output_path = os.path.join(self.output_dir, f'{self.tag}.{filename}')

        with open(output_path, 'w') as f:
            f.write(data_str)
        
    def save_stats(self, filename, threshold, num_bins=10):
        num_additions_static = output_static(self.data_num_additions, threshold, num_bins)
        output_path = os.path.join(self.output_dir, f'{self.tag}.{filename}.yaml')


        with open(output_path, 'w') as f:
            yaml.dump(num_additions_static, f, allow_unicode=True, sort_keys=False)

    def save_numpy(self, filename):
        output_path = os.path.join(self.output_dir, f'{self.tag}.{filename}')


        np.save(output_path, self.data_num_additions)

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--input_dir", type=str, default="./data/diff_dataset", help="Experiment dump path")

    parser.add_argument("--output_dir", type=str, default="./dumped", help="Experiment dump path")
    parser.add_argument("--testset_only", action='store_true', default=False)
    parser.add_argument("--buchberger_threshold", type=int, default=10000)
    parser.add_argument("--max_itrs", type=int, default=100)
    parser.add_argument("--field", type=str)
    parser.add_argument("--nvars", type=int, default=3)
    parser.add_argument("--matrix_entry_lower_bound", type=int, default=-1000)
    parser.add_argument("--matrix_entry_upper_bound", type=int, default=1000)
    parser.add_argument("--matrix_entry_distribution", type=str, default='uniform')
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--lower_lim_for_num_tokens", type=int, default=0, help="このトークン数未満のデータは無視する")
    parser.add_argument("--upper_lim_for_num_tokens", type=int, default=None, help="このトークン数より大きいデータは無視する")
    parser.add_argument("--test_num_matrices_per_F", type=int, default=1)
    parser.add_argument("--train_num_matrices_per_F", type=int, default=1)
    parser.add_argument("--test_num_samples", type=int)
    parser.add_argument("--train_num_samples", type=int)


    return parser 

def main():
    parser = get_parser()
    params = parser.parse_args()

    nvars = params.nvars
    threshold = params.buchberger_threshold
    max_itrs = params.max_itrs
    matrix_entry_lower_bound = params.matrix_entry_lower_bound
    matrix_entry_upper_bound = params.matrix_entry_upper_bound
    matrix_entry_distribution = params.matrix_entry_distribution
    lower_lim_for_num_tokens = params.lower_lim_for_num_tokens
    upper_lim_for_num_tokens = params.upper_lim_for_num_tokens
    input_dir = params.input_dir
    output_dir = params.output_dir
    test_num_matrices_per_F = params.test_num_matrices_per_F
    train_num_matrices_per_F = params.train_num_matrices_per_F
    test_num_samples = params.test_num_samples
    train_num_samples = params.train_num_samples

    n_jobs = -1

    field_name = params.field
    if field_name == 'QQ':
        field = QQ
    elif field_name == 'RR':
        field = RR
    elif field_name == 'ZZ':
        field = ZZ
    elif field_name[:2] == 'GF':
        order = int(field_name[2:])
        field = GF(order)
    
    base_name =f'data_{field_name}_n={nvars}'
    
    # ----------------test----------------
    input_filename = base_name + '.test.lex.infix'
    F_G_str_list = load_data(input_dir, filename=input_filename, num_samples=test_num_samples)
    build_dataset = BuildDataset(tag = 'test',
                                 F_G_str_list = F_G_str_list, 
                                 num_matrices_per_F =test_num_matrices_per_F,
                                 nvars = nvars, 
                                 field = field, 
                                 max_itrs = max_itrs, 
                                 threshold = threshold, 
                                 matrix_entry_lower_bound = matrix_entry_lower_bound, 
                                 matrix_entry_upper_bound = matrix_entry_upper_bound, 
                                 matrix_entry_distribution = matrix_entry_distribution,
                                 lower_lim_for_num_tokens = lower_lim_for_num_tokens, 
                                 upper_lim_for_num_tokens = upper_lim_for_num_tokens, 
                                 n_jobs=n_jobs)
    
    data_str_list, data_num_additions_list = build_dataset()

    test_writer = Writer(data_str_list, data_num_additions_list, output_dir, 'test')
    test_writer.write(filename='F_matrix_nadds')
    test_writer.save_stats(filename='nadds_static', threshold=threshold, num_bins=params.num_bins)
    test_writer.save_numpy(filename='nadds.npy')

    # ----------------train----------------

    if not params.testset_only:
        input_filename = base_name + '.train.lex.infix'
        F_G_str_list = load_data(input_dir, filename=input_filename, num_samples=train_num_samples)

        build_dataset = BuildDataset(tag = 'train',
                                     F_G_str_list = F_G_str_list, 
                                     num_matrices_per_F = train_num_matrices_per_F,
                                     nvars = nvars, 
                                     field = field, 
                                     max_itrs = max_itrs, 
                                     threshold = threshold, 
                                     matrix_entry_lower_bound = matrix_entry_lower_bound, 
                                     matrix_entry_upper_bound = matrix_entry_upper_bound, 
                                     matrix_entry_distribution = matrix_entry_distribution,
                                     lower_lim_for_num_tokens = lower_lim_for_num_tokens, 
                                     upper_lim_for_num_tokens = upper_lim_for_num_tokens, 
                                     n_jobs=n_jobs)

        data_str_list, data_num_additions_list = build_dataset()

        test_writer = Writer(data_str_list, data_num_additions_list, output_dir, 'train')
        test_writer.write(filename='F_matrix_nadds')
        test_writer.save_stats(filename='nadds_static', threshold=threshold, num_bins=params.num_bins)
        test_writer.save_numpy(filename='nadds.npy')

    print('done!')

if __name__ == '__main__':
    main()

    