class LaplaceProblemDataset(Dataset):
    def __init__(self, input_file, n):
        data = pd.read_csv(input_file, sep=' ', header=None, nrows=n).to_numpy()
    
        self.samples = []
        
        for sample in data:
            input_coefficients = sample[:16]
            value = sample[16:]
            self.samples.append({'coefficients': input_coefficients, 'value': value})
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
