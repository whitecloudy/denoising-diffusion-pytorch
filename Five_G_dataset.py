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
    def __init__(self, data_path, self_normalize=True, return_complex=True, real_dim=None, transpose=None):
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

        self.transpose = transpose

        self.self_normalize = self_normalize
        self.return_complex = return_complex
        self.real_dim = real_dim

        # if dtype is None:
        #     if return_complex:
        #         self.dtype = torch.complex64
        #     else:
        #         self.dtype = torch.float32

    @staticmethod
    def complex_to_real(data : torch.tensor, dim : int):
        """
        Convert complex data to real data.
        Args:
            data (torch.tensor): Complex data (Time slot, Node, Subcarrier)
        Returns:
            real_data (torch.tensor): Real data (Time slot, Node*2, Subcarrier)
        """
        assert type(dim) == int, "The dimension should be an integer"
        data = torch.view_as_complex(data)
        real_data = torch.cat((data.real, data.imag), dim=dim)
        return real_data
    
    @staticmethod
    def real_to_complex(data : torch.tensor, dim : int):
        """
        Convert real data to complex data.
        Args:
            data (torch.tensor): Real data (Time slot, Node*2, Subcarrier)
        Returns:
            complex_data (torch.tensor): Complex data (Time slot, Node, Subcarrier)
        """
        assert data.shape[dim] % 2 == 0, "The first dimension of the data should be even"
        assert type(dim) == int, "The dimension should be an integer"
        # Split the data into real and imaginary parts
        real_imag_data = torch.split(data, data.shape[dim] // 2, dim=dim)
        # Concatenate the real and imaginary parts to form complex data
        complex_data = torch.view_as_real(real_imag_data[0] + 1j * real_imag_data[1])
        return complex_data
    
    @staticmethod
    def normalize(data : torch.tensor, cond : torch.tensor):
        """
        Normalize the data and condition.
        Args:
            data (torch.tensor): Data (Time slot, Node, Subcarrier)
            cond (torch.tensor): Condition (Time slot, Node, Subcarrier)
        Returns:
            normalized_data (torch.tensor): Normalized data
            normalized_cond (torch.tensor): Normalized condition
        """
        cond_std = cond.std()
        normalized_data = data / cond_std
        normalized_cond = cond / cond_std
        return normalized_data, normalized_cond

    def __len__(self):
        return len(self.data_filenames)

    @torch.inference_mode()
    def __getitem__(self, idx):
        filename = self.data_filenames.iloc[idx]["filename"]
        # (Time slot, Node, Subcarrier)

        with np.load(filename) as loaded_data:
            # (Time slot, Node, Subcarrier)
            data = loaded_data['data'].astype(np.complex64)
            cond = loaded_data['cond'].astype(np.complex64)
            if self.transpose is not None:
                data = np.transpose(data, self.transpose)
                cond = np.transpose(cond, self.transpose)
            data = torch.from_numpy(data)
            cond = torch.from_numpy(cond)

        if self.self_normalize:
            data, cond = self.normalize(data, cond)

        data = torch.view_as_real(data)
        cond = torch.view_as_real(cond)

        if not self.return_complex:
            data = self.complex_to_real(data, self.real_dim)
            cond = self.complex_to_real(cond, self.real_dim)

        return data, cond
    
if __name__ == "__main__":
    test_dataset = Five_G_dataset(["../ssddata/RENEW/ArgosCSI-96x8-2016-05-01-06-38-03-2.4GHz-static"], return_complex=True)
    from torch.utils.data import DataLoader
    from multiprocessing import cpu_count
    from accelerate import Accelerator
    import tqdm
    dl = DataLoader(test_dataset, batch_size = 128, shuffle = False, pin_memory = True, num_workers = cpu_count())
    for i, (data, cond) in enumerate(dl):
        print(data.shape)
        reald = test_dataset.complex_to_real(data)
        print(reald.shape)
        back2c = test_dataset.real_to_complex(reald)
        print(back2c.shape)
        print(data==back2c)
        break