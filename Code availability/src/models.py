import torch
import torch.nn as nn
import torch.utils.data as data


activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
}


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="relu"):
        super(MLP, self).__init__()
        cs = input_size
        self.hidden_layers = hidden_sizes
        self.hls = nn.Sequential()
        for i, hs in enumerate(hidden_sizes):
            self.hls.add_module(f"hl_{i}", nn.Linear(cs, hs, dtype=torch.float64))
            self.hls.add_module(f"activ_{i}", activations[activation])
            cs = hs
        self.fc2 = nn.Linear(cs, output_size, dtype=torch.float64)

    def forward(self, x):
        x.to(torch.float32)
        out = self.hls(x)
        out = self.fc2(out)
        return out


class MyDataset(data.Dataset):
    def __init__(self, df, recipe_indexes):
        rec_cols = []
        for i in recipe_indexes:
            rec_cols += [f"r_{i}", f"g_{i}", f"b_{i}"]
        self.x_data = df[rec_cols].values / 255
        self.y_data = df[["conc_water", "conc_co2", "conc_nh3"]].values

        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    # for (x, y) in tqdm(iterator, desc="Training", leave=False):
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        # for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs