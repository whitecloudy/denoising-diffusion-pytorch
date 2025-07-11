import numpy as np
import pandas as pd
import os
import torch
from glob import glob
from torch.utils.data import Dataset

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


def np_split_in_size(data, size, axis=0):
    """
    Split the data into chunks of a given size along a specified axis.
    Args:
        data (torch.tensor): Data to be split.
        size (int): Size of each chunk.
        axis (int): Axis along which to split the data.
    Returns:
        list: List of chunks of the data.
    """

    splited_list = np.split(data, np.arange(size, data.shape[axis], size), axis=axis)
    if len(splited_list[-1]) < size:
        splited_list = splited_list[:-1]
    return splited_list

class Five_G_singlefile_dataset(Dataset):
    """
    5G Dataset for time series prediction.
    Args:
        data_path (str): Path to the npz file containing the 5G dataset.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        complex data (torch.tensor): Uplink data(Node, Time slot, Subcarrier), Downlink data(Node, Time slot, Subcarrier)
    """
    def __init__(self, data_path, time_node_shape=(14, 16), self_normalize=True, return_complex=True, real_dim=None, transpose=None):
        super().__init__()
        np_datas = []
        print(f"Loading data from {data_path}...")

        def single_file_handle(path):
            loaded_data = np.load(path).astype(np.complex64)
            loaded_data = np.transpose(loaded_data, (1, 0, 2, 3))
            #  (N, T, Client, Splitted Time slot, Splited Node, Subcarrier)
            loaded_data = self.time_node_spliter(loaded_data, split_time_node=time_node_shape)

            # loaded_data = loaded_data[:, :, 1]  # TEMP : only use the second client
            loaded_data = loaded_data[1]  # TEMP : only use the second node

            return loaded_data.reshape(-1, *loaded_data.shape[-3:])

        # (Time slot, client, Node, Subcarrier)
        # T, 8, 96, 52
        if isinstance(data_path, list):
            for path in data_path:
                # self.filenames += glob(f'{path}/**/*.mat', recursive=True)
                data = single_file_handle(path)
                print(data.shape)
                np_datas.append(data)
        elif isinstance(data_path, str):
            data = single_file_handle(data_path)
            np_datas.append(data)
        else:
            raise ValueError("data_path should be a string or a list of strings")
        np_datas = np.concatenate(np_datas, axis=0)
        np_datas = np_datas.reshape(-1, *np_datas.shape[-3:])  # (N, Time slot, Node, Subcarrier)
        
        self.transpose = transpose

        self.data = np_datas[:,:,:,:26]
        self.cond = np_datas[:,:,:,26:]

        self.self_normalize = self_normalize
        self.return_complex = return_complex
        self.real_dim = real_dim

        del np_datas
        # if dtype is None:
        #     if return_complex:
        #         self.dtype = torch.complex64
        #     else:
        #         self.dtype = torch.float32

    @staticmethod
    def time_node_spliter(data, split_time_node=(14, 8)):
        """
        Split the data into training and validation sets.
        Args:
            data (torch.tensor): Data (Time slot, Node, Subcarrier)
            split_time_node (tuple): Tuple of two integers, the first one is the number of time slots for training,
                                    the second one is the number of nodes for training.
        Returns:
            split_data (torch.tensor): Split data (N, T, Client, Splitted Time slot, Splited Node, Subcarrier)
        """
        # (Client, Time slot, Node, Subcarrier)
        time_split = np_split_in_size(data, split_time_node[0], axis=1)
        # if the last time slot is not fulled, remove it

        time_split = np.array(time_split)  
        # time_split = (T, Client, Splitted Time slot, Node, Subcarrier)
        time_node_split = np_split_in_size(time_split, split_time_node[1], axis=3)
        
        # (N, T, Client, Splitted Time slot, Splited Node, Subcarrier)
        time_node_split = np.stack(time_node_split, axis=0)

        return time_node_split

    def __len__(self):
        return self.data.shape[0]

    @torch.inference_mode()
    def __getitem__(self, idx):
        # (Time slot, Node, Subcarrier)
        data = self.data[idx]
        cond = self.cond[idx]

        if self.transpose is not None:
            data = np.transpose(data, self.transpose)
            cond = np.transpose(cond, self.transpose)
        data = torch.from_numpy(data)
        cond = torch.from_numpy(cond)

        if self.self_normalize:
            data, cond = normalize(data, cond)

        data = torch.view_as_real(data)
        cond = torch.view_as_real(cond)

        if not self.return_complex:
            data = complex_to_real(data, self.real_dim)
            cond = complex_to_real(cond, self.real_dim)

        return data, cond


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
            data, cond = normalize(data, cond)

        data = torch.view_as_real(data)
        cond = torch.view_as_real(cond)

        if not self.return_complex:
            data = complex_to_real(data, self.real_dim)
            cond = complex_to_real(cond, self.real_dim)

        return data, cond
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from multiprocessing import cpu_count
    from accelerate import Accelerator
    import tqdm
    
    training_datafiles = ['../data/RENEW_processed/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS.npy',
                            '../data/RENEW_processed/ArgosCSI-96x8-2016-05-01-06-57-58-2.4GHz-continuousmobile.npy',
                            '../data/RENEW_processed/ArgosCSI-96x2-2016-12-07-03-00-36_rotation_mob_horizontal_omni.npy']

    dataset = Five_G_singlefile_dataset(data_path=training_datafiles, transpose=(0, 2, 1))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=cpu_count(), pin_memory=True)

    input("Press Enter to continue...")

    for data, cond in dataloader:
        print(data.shape, cond.shape)
        print(data.dtype, cond.dtype)
        print(data[0, 0, 0], cond[0, 0, 0])
        print(data[0, 1, 0], cond[0, 1, 0])
        print(data[0, 2, 0], cond[0, 2, 0])
        break