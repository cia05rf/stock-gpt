import torch

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 30 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data loading
def get_batch(split, train_data, val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    idx = torch.randperm(data.shape[0])
    data = data[idx]
    x = data[:, :block_size]
    y = data[:, 1:block_size+1]
    ix = torch.randint(len(data), (batch_size,))
    x = x[ix,:]
    y = y[ix,:]
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    # x = torch.stack([data[i:i+block_size] for i in ix])
    # y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y