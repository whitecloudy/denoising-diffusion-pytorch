import numpy as np
import pandas as pd
import os
import torch
from glob import glob
from torch.utils.data import Dataset

class Five_G_dataset(Dataset):
    """
    5G Dataset for time series prediction.
    Args:
        data_path (str): Path to the npz file containing the 5G dataset.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        complex data (torch.tensor): Uplink data(Node, Time slot, Subcarrier), Downlink data(Node, Time slot, Subcarrier)
    """
    def __init__(self, data_path, return_complex=True, transform=None):
        super().__init__()
        self.data_filenames = []

        if isinstance(data_path, list):
            for path in data_path:
                # self.filenames += glob(f'{path}/**/*.mat', recursive=True)
                self.data_filenames += glob(f'{path}/**/*.npz', recursive=True)
        elif isinstance(data_path, str):
            self.data_filenames += glob(f'{data_path}/**/*.npz', recursive=True)
        else:
            raise ValueError("data_path should be a string or a list of strings")

        self.data_filenames = pd.DataFrame(self.data_filenames, columns=['filename'])

        self.transform = transform
        self.return_complex = return_complex

        # if dtype is None:
        #     if return_complex:
        #         self.dtype = torch.complex64
        #     else:
        #         self.dtype = torch.float32

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        filename = self.data_filenames.iloc[idx]["filename"]
        # (Time slot, Node, Subcarrier)
        loaded_data = np.load(filename)
        
        # (Node, Time slot, Subcarrier)
        data = np.transpose(loaded_data['data'].astype(np.complex64), (1, 0, 2))
        cond = np.transpose(loaded_data['cond'].astype(np.complex64), (1, 0, 2))

        if self.return_complex:
            return torch.from_numpy(data), torch.from_numpy(cond)
        else:
            data = np.concatenate((data.real, data.imag), axis=0)
            cond = np.concatenate((cond.real, cond.imag), axis=0)
            return torch.from_numpy(data), torch.from_numpy(cond)
    
if __name__ == "__main__":
    test_dataset = Five_G_dataset(["../ssddata/RENEW/ArgosCSI-96x8-2016-05-01-06-38-03-2.4GHz-static"], return_complex=False)
    from torch.utils.data import DataLoader
    from multiprocessing import cpu_count
    from accelerate import Accelerator
    import tqdm
    dl = DataLoader(test_dataset, batch_size = 128, shuffle = True, pin_memory = True, num_workers = cpu_count())
    accelerator = Accelerator(
        mixed_precision = 'no'
    )
    dl = accelerator.prepare(dl)

    for data, cond in tqdm.tqdm(dl, total=len(dl)):
        if torch.any(torch.isnan(data)):
            print("Data contains NaN values")
        if torch.any(torch.isnan(cond)):
            print("Condition contains NaN values")