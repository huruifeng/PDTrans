'''
### Dx model design:
* DA. the input data is gene expression data of current visit, current UPDRS, gene expression data of 1 previous visit, 1 previous UPDRS, the output data is UPDRS of next visit
  * -- the model is to predict the UPDRS of next visit
  * -- current expression data + current UPDRS + 1 previous expression data + 1 previous UPDRS -> next UPDRS

* DB. the input data is gene expression data of current visit, current UPDRS, gene expression data of 1 previous visit, 1 previous UPDRS, time period until next visit, the output data is UPDRS of next visit
  * -- the model is to predict the UPDRS of next visit
  * -- current expression data + current UPDRS + 1 previous expression data + 1 previous UPDRS + time period -> next UPDRS
* <u>DB is the most general model design for Dx model, and it can be used for both DA and DB by setting the time period as 0 for DA </u>
  * Prepare a data table with columns: expression data, current UPDRS, 1 previous expression data, 1 previous UPDRS, time period until next visit, UPDRS of next visit
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, genes_time_series, updrs_time_series):
        self.genes_time_series = genes_time_series  # Shape: (num_samples, time_points, num_genes)
        self.updrs_time_series = updrs_time_series  # Shape: (num_samples, time_points)

    def __len__(self):
        return len(self.updrs_time_series)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.genes_time_series[idx], dtype=torch.float32),
            torch.tensor(self.updrs_time_series[idx, :-1], dtype=torch.float32),  # Inputs: all but last
            torch.tensor(self.updrs_time_series[idx, -1], dtype=torch.float32),   # Target: last time point
        )

# Define the BiLSTM model
class BiLSTMPredictor(nn.Module):
    def __init__(self, num_genes, hidden_size=128, num_layers=2):
        super(BiLSTMPredictor, self).__init__()

        self.lstm_genes = nn.LSTM(
            input_size=num_genes,  # Gene data at each time point
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

        self.lstm_updrs = nn.LSTM(
            input_size=1,  # UPDRS at each time point
            hidden_size=1,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

        self.fc_genes = nn.Linear(hidden_size, hidden_size)
        self.fc_updrs = nn.Linear(1, 1)

        self.fc = nn.Linear(hidden_size +1, 16)

        self.output_layer = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, genes_time_series, updrs_time_series):
        genes_output, _ = self.lstm_genes(genes_time_series)
        updrs_output, _ = self.lstm_updrs(updrs_time_series)

        genes_output = genes_output[:, -1, :]   # Extract only the last time step
        updrs_output = updrs_output[:, -1, :]   # Get the last time point


        genes_output = self.relu(self.fc_genes(genes_output))
        updrs_output = self.relu(self.fc_updrs(updrs_output))

        genes_output = genes_output.unsqueeze(1)
        updrs_output = updrs_output.unsqueeze(1)

        x = torch.cat([genes_output, updrs_output], dim=-1)
        x = self.relu(self.fc(x))
        x = self.output_layer(x)
        x = x.squeeze(1)

        return x

# Training function
def train_model(model, train_loader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    run_len = len(train_loader) // 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        all_targets = []
        all_predictions = []

        print(f"Epoch %{len(str(num_epochs))}d/%d " % (epoch + 1, num_epochs), end="", flush=True)
        for i, (genes_time_series, updrs_time_series, target_updrs) in enumerate(train_loader,0):
            genes_time_series = genes_time_series.to(device)
            updrs_time_series = updrs_time_series.to(device)
            target_updrs = target_updrs.to(device)

            # Forward pass
            outputs = model(genes_time_series, updrs_time_series)
            loss = criterion(outputs, target_updrs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            all_targets.extend(target_updrs.cpu().numpy())
            all_predictions.extend(outputs.cpu().detach().numpy())

            if i % run_len == 0:
                print("#", end="", flush=True)
            # Calculate R² score for the epoch
        r2 = r2_score(all_targets, all_predictions)
        print(f" Loss: {epoch_loss / len(train_loader):>.4f}, R²: {r2:.4f}", end="", flush=True)
        print()

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("../results/training_testing/PPMI_data_previous_current_next.csv", index_col=0)
    ## remove rows if THERE IS MISSING DATA
    df = df.dropna(axis=0)

    num_genes = 874
    num_samples = df.shape[0]

    time_points = 3

    gene_expression_t1 = df.loc[:, df.columns.str.startswith('previous_ENSG')]
    gene_expression_t2 = df.loc[:, df.columns.str.startswith('current_ENSG')]

    updrs_t1 = df.loc[:, df.columns.str.startswith('previous_updrs')]
    updrs_t2 = df.loc[:, df.columns.str.startswith('current_updrs')]
    updrs_t3 = df.loc[:, df.columns.str.startswith('next_updrs')]

    genes_time_series = np.zeros((num_samples, time_points-1, num_genes)) # Shape: (num_samples, time_points, num_genes)
    updrs_time_series = np.zeros((num_samples, time_points, 1)) # Shape: (num_samples, time_points, 1)

    for i in range(num_samples):
        genes_time_series[i, 0] = gene_expression_t1.iloc[i].values
        genes_time_series[i, 1] = gene_expression_t2.iloc[i].values

        updrs_time_series[i, 0] = updrs_t1.iloc[i].values
        updrs_time_series[i, 1] = updrs_t2.iloc[i].values
        updrs_time_series[i, 2] = updrs_t3.iloc[i].values

    dataset = UPDRSDataset(genes_time_series, updrs_time_series)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMPredictor(num_genes=num_genes)
    train_model(model, dataloader, num_epochs=100, learning_rate=1e-3, device=device)

    ## Save the model
    torch.save(model.state_dict(), '../models/model_D_BiLSTM_model.pt')

