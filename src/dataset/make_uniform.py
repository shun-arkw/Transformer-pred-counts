import os
import time
import argparse
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from collections import defaultdict


def load_data(input_dir, filename):
    input_path = os.path.join(input_dir, filename)
    with open(input_path, 'r') as file:
        contents = file.read()

    matrix_F_nadds_list = contents.split('\n') 

    return matrix_F_nadds_list


class MakeUniformDataset():
    def __init__(self, bins, num_samples):
        self.bins = bins
        self.max_samples = num_samples
        self.num_samples_in_each_class = defaultdict(lambda: 0)

    def __call__(self, matrix_F_nadds_list):

        data_str_list = []
        data_num_additions_list = []
        for matrix_F_nadds_str in matrix_F_nadds_list:
            bins = self.bins
            num_classes = len(bins) - 1

            _, nadds = self.preprocess_for_nadds(matrix_F_nadds_str)

            class_id = np.digitize(nadds, bins=self.bins) - 1 # class_id = 0, 1, 2, ..., num_classes - 1
            if 0 <= class_id < num_classes:
                if self.num_samples_in_each_class[class_id] < self.max_samples:
                    self.num_samples_in_each_class[class_id] += 1
                    data_str_list.append(matrix_F_nadds_str)
                    data_num_additions_list.append(nadds)

            if all([num_samples == self.max_samples for num_samples in self.num_samples_in_each_class.values()]):
                break

        return data_str_list, data_num_additions_list
         
    def preprocess_for_nadds(self, matrix_F_nadds_str):
        nadds_str = matrix_F_nadds_str.split(':')[-1]
        nadds_str = nadds_str.strip()
        nadds = int(nadds_str)

        return matrix_F_nadds_str, nadds



class Writer():
    def __init__(self, data_str_list, data_num_additions_list, output_dir, bins, tag):

        self.data_str_list = data_str_list
        self.data_num_additions = np.array(data_num_additions_list)
        self.output_dir = output_dir
        self.tag = tag
        self.bins = bins

    def write(self, filename):
        data_str = '\n'.join(self.data_str_list)
        output_path = os.path.join(self.output_dir, f'{self.tag}.{filename}')

        with open(output_path, 'w') as f:
            f.write(data_str)
        
    def save_stats(self, filename):
        num_additions_static = self.output_static()
        output_path = os.path.join(self.output_dir, f'{self.tag}.{filename}.yaml')

        with open(output_path, 'w') as f:
            yaml.dump(num_additions_static, f, allow_unicode=True, sort_keys=False)

    def output_static(self):
        data_nadds = self.data_num_additions
        nadds_static = {}
        num_samples = len(data_nadds)
        nadds_static['mean'] = float(np.mean(data_nadds))
        nadds_static['std'] = float(np.std(data_nadds))
        nadds_static['max'] = float(np.max(data_nadds))
        nadds_static['min'] = float(np.min(data_nadds))
        nadds_static['median'] = float(np.median(data_nadds))
        nadds_static['num_samples'] = int(num_samples)

        num_bins = len(self.bins) - 1
        ret = plt.hist(data_nadds, bins=num_bins, range=(0, self.bins[-1]), density=False)  
        counts, bin_edges = ret[0], ret[1]
        counts = counts.astype(int)

        for i in range(num_bins):
            nadds_static[f'{bin_edges[i]} <= num_additions < {bin_edges[i+1]}'] = int(counts[i])

        nadds_static[f'{bin_edges[num_bins]} <= num_additions'] = int(num_samples - np.sum(counts))

        return nadds_static
    
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
    parser.add_argument("--field", type=str)
    parser.add_argument("--nvars", type=int, default=3)
    parser.add_argument("--train_num_samples", type=int)
    parser.add_argument("--test_num_samples", type=int)

    return parser 

def main():

    
    parser = get_parser()
    params = parser.parse_args()
    input_dir = params.input_dir
    output_dir = params.output_dir
    test_num_samples = params.test_num_samples
    train_num_samples = params.train_num_samples

    bins = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    
    # ----------------test----------------
    input_filename = 'test.F_matrix_nadds'
    matrix_F_nadds_list = load_data(input_dir, filename=input_filename)
   
    build_dataset = MakeUniformDataset(bins=bins, num_samples=test_num_samples)

    data_str_list, data_num_additions_list = build_dataset(matrix_F_nadds_list=matrix_F_nadds_list)

    test_writer = Writer(data_str_list, data_num_additions_list, output_dir, bins, 'test')
    test_writer.write(filename='uni_F_matrix_nadds')
    test_writer.save_stats(filename='uni_nadds_static')


    # # ----------------train----------------

    if not params.testset_only:
        input_filename = 'train.F_matrix_nadds'
        matrix_F_nadds_list = load_data(input_dir, filename=input_filename)
    
        build_dataset = MakeUniformDataset(bins=bins, num_samples=train_num_samples)

        data_str_list, data_num_additions_list = build_dataset(matrix_F_nadds_list=matrix_F_nadds_list)

        test_writer = Writer(data_str_list, data_num_additions_list, output_dir, bins, 'train')
        test_writer.write(filename='uni_F_matrix_nadds')
        test_writer.save_stats(filename='uni_nadds_static')

    print('done!')

if __name__ == '__main__':
    main()
