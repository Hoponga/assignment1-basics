import numpy as np 
import torch 
import numpy.typing as npt


def get_datapoints_from_source(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = torch.stack([torch.from_numpy(dataset[s : s + context_length].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(dataset[s + 1 : s + context_length + 1].astype(np.int64)) for s in starts])
    return x.to(device), y.to(device) 




def save_checkpoint(): 




def load_checkpoint(): 
    