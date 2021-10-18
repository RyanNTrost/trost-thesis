import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import random
from scipy.fftpack import dct, idct


class LaplaceProblemDataset(Dataset):
    def __init__(self, input_file, n):
        grid_size = int(input_file.split('_')[1].split('x')[0])
        grid_size_squared = grid_size**2

        data = pd.read_csv(input_file, sep=' ', header=None, nrows=n).to_numpy()
    
        self.samples = []
        
        for sample in data:
            input_coefficients = sample[:grid_size_squared]
            input_2d = np.reshape(input_coefficients, (-1, 2))
            dct_input = dct2(input_2d).reshape(-1)
            
            output_values = sample[grid_size_squared:]
            output_2d = np.reshape(output_values, (-1, 2))
            dct_output = dct2(output_2d).reshape(-1)

            self.samples.append({
                'coefficients': input_coefficients, 
                'coefficients-dct': dct_input, 
                'output': output_values,
                'output-dct': dct_output
            })
            
    def __len__(self):
        return len(self.samples)

    def input_size(self):
        return len(self.samples[0]['coefficients'])
    
    def output_size(self):
        return len(self.samples[0]['output'])
    
    def __getitem__(self, idx):
        return self.samples[idx]


def dct2(vector):
    return dct(dct(vector.T, norm='ortho').T, norm='ortho')


def idct2(vector):
    return idct(idct(vector.T, norm='ortho').T, norm='ortho')


def split_data(dataset, train_split=0.5, validation_split=0.2, test_split=0.3):
    assert(train_split + validation_split + test_split == 1)

    reset_random()
    
    train_size = int(train_split * len(dataset))
    validation_size = int(validation_split * len(dataset))
    test_size = int(test_split * len(dataset))

    training_data, validation_data, test_data = random_split(dataset, [train_size, validation_size, test_size])

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader


def reset_random():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)