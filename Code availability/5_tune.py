import pandas as pd

import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch

import time
import tempfile

import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import copy, yaml
import wandb

from src.models import MLP, train, evaluate, epoch_time, MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 1
torch.manual_seed(seed)

wandb.init()

select = wandb.config.select
hidden_size_1 = wandb.config.hidden_size_1
hidden_size_2 = wandb.config.hidden_size_2
activation = wandb.config.activation
lr = wandb.config.lr
weight_decay = wandb.config.weight_decay
epochs = 100000

    # %%

recipes = list(range(10))
dataset_train = MyDataset(pd.read_csv(f"data/select/{select}_train.csv"), recipes)
dataset_test = MyDataset(pd.read_csv(f"data/select/{select}_test.csv"), recipes)
test_size = int(0.5 * len(dataset_test))
test_data, valid_data = torch.utils.data.random_split(
    dataset_test, [len(dataset_test) - test_size, test_size]
)
BATCH_SIZE = 256
train_iterator = data.DataLoader(dataset_train,shuffle=True,batch_size=BATCH_SIZE)
valid_iterator = data.DataLoader(valid_data,batch_size=BATCH_SIZE)
test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE)

# %%
model = MLP(3 * len(recipes), [hidden_size_1 * 5, hidden_size_2 * 5], 3, activation).to(device)
optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)
criterion = nn.L1Loss()
criterion = criterion.to(device)


best_valid_loss = float('inf')

for epoch in range(epochs):

    start_time = time.monotonic()

    train_loss = train(model, train_iterator, optimizer, criterion, device)
    valid_loss = evaluate(model, valid_iterator, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'results/models/{select}.pt')
        best_model = copy.deepcopy(model)

    wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "best_valid_loss": best_valid_loss})

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.5f} | Val. Loss: {valid_loss:.5f}')

wandb.log({"test_loss": evaluate(best_model, test_iterator, criterion, device)})
# torch.save(best_model.state_dict(), f"models/{select}.pt")
# wandb.save(f"models/{select}.pt")
# %%
