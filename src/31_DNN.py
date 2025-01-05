import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

## Build a deep neural network
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x

def train_model(model, train_loader, val_loader=None, epoch=30, device="cpu"):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss = 0.0

    for epoch in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    if val_loader is not None:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        print("Train loss: {:.4f}, Val loss: {:.4f}".format(train_loss/len(train_loader), val_loss/len(val_loader)))
    else:
        print("Train loss: {:.4f}".format(train_loss/len(train_loader)))

    return train_loss

def test_model(model, test_loader, device="cpu"):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0

    print("Testing...")

    test_results_df = pd.DataFrame(columns=["Actual", "Predicted"])

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            test_results_df = pd.concat([test_results_df, pd.DataFrame(
                {"Actual": targets.squeeze().cpu().numpy(), "Predicted": outputs.squeeze().cpu().numpy()})], ignore_index=True)

    return test_results_df



if __name__ == "__main__":
    df = pd.read_csv("../results/training_testing/PPMI_data_current_next.csv", index_col=0)
    ## remove rows if THERE IS MISSING DATA
    df = df.dropna(axis=0)
    df = df.loc[df["time_period"] == 12, :]

    val_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(val_df.index)

    x_train = train_df.loc[:, train_df.columns.str.startswith("ENSG") | train_df.columns.str.startswith("current_updrs")].values
    y_train = train_df.loc[:, train_df.columns.str.startswith("next_updrs")].values

    x_val = val_df.loc[:, val_df.columns.str.startswith("ENSG") | val_df.columns.str.startswith("current_updrs")].values
    y_val = val_df.loc[:, val_df.columns.str.startswith("next_updrs")].values

    ## original r2 and pcc
    r2 = r2_score(train_df["current_updrs"].values, train_df["next_updrs"].values)
    pcc = pearsonr(train_df["current_updrs"].values, train_df["next_updrs"].values)[0]
    print(f"Original train: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    r2 = r2_score(val_df["current_updrs"].values, val_df["next_updrs"].values)
    pcc = pearsonr(val_df["current_updrs"].values, val_df["next_updrs"].values)[0]
    print(f"Original val: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    # Define model
    model = DNN(input_size=x_train.shape[1], hidden_size=64, output_size=y_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train model
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    train_model(model, train_loader, val_loader, epoch=30, device=device)


   ## predictions
    test_df = pd.read_csv("../results/training_testing/PDBP_data_current_next.csv", index_col=0)
    test_df = test_df.dropna(axis=0)

    x_test = test_df.loc[:, test_df.columns.str.startswith("ENSG") | test_df.columns.str.startswith("current_updrs")].values
    y_test =test_df.loc[:, test_df.columns.str.startswith("next_updrs")].values

    r2 = r2_score(test_df["current_updrs"].values, test_df["next_updrs"].values)
    pcc = pearsonr(test_df["current_updrs"].values, test_df["next_updrs"].values)[0]
    print(f"Original test: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    test_results_df = test_model(model, test_loader, device)

    ## Calculate R^2
    r2 = r2_score(test_results_df["Actual"], test_results_df["Predicted"])
    pcc = pearsonr(test_results_df["Actual"], test_results_df["Predicted"])[0]
    print(f"R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    # %%
    ## plot actual vs predicted
    plt.scatter(test_results_df["Actual"], test_results_df["Predicted"])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

