'''
### Bx model design:
* BA. the input data is gene expression data of current visit and current UPDRS, the output data is UPDRS of next visit (in 12 months)
  * -- the model is to predict the UPDRS of next visit
  * -- current expression data + current UPDRS -> next UPDRS in 12 months

* BB. the input data is gene expression data of current visit, current UPDRS, time period until next visit, the output data is UPDRS of next visit
  * -- the model is to predict the UPDRS of next visit
  * -- current expression data + current UPDRS + time period -> next UPDRS
* <u>BB is the most general model design for Bx model, and it can be used for both BA and BB by setting the time period as 12 for BA</u>
  * Prepare a data table with columns: current expression data, current UPDRS, time period until next visit, UPDRS of next visit
'''
import pandas as pd

'''
Build a model to predict the UPDRS of next visit (in 12 months), This is a regression model
Input: 
    - current expression data (874 genes transcriptome data),
    - current UPDRS (continuous variable), 
    - time period until next visit (integer variable)
Output:
    - UPDRS of next visit
    
Use PyTorch to build the model class, and use PyTorch to train the model, implement the model using transformer blocks
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, expression_data, current_updrs, time_periods, next_updrs):
        self.expression_data = expression_data
        self.current_updrs = current_updrs
        self.time_periods = time_periods
        self.next_updrs = next_updrs

    def __len__(self):
        return len(self.next_updrs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.expression_data[idx], dtype=torch.float32),
            torch.tensor(self.current_updrs[idx], dtype=torch.float32),
            torch.tensor(self.time_periods[idx], dtype=torch.float32),
            torch.tensor(self.next_updrs[idx], dtype=torch.float32),
        )

# Define the Transformer-based model
class UPDRSTransformer(nn.Module):
    def __init__(self, num_genes, d_model=128, nhead=8, num_layers=12, dropout=0.1):
        super(UPDRSTransformer, self).__init__()
        self.input_layer = nn.Linear(num_genes, d_model)  # +2 for current UPDRS and time period
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.combined_layer = nn.Linear(d_model + 2, 16)
        self.output_layer = nn.Linear(16, 1)  # Single output for regression
        self.relu = nn.ReLU()

    def forward(self, expression_data, current_updrs, time_periods):
        # Concatenate inputs
        # combined_input = torch.cat([expression_data, current_updrs.unsqueeze(1), time_periods.unsqueeze(1)], dim=1)
        combined_input = torch.cat([expression_data], dim=1)
        # Pass through input layer
        x = self.relu(self.input_layer(combined_input))

        # Add dummy sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, d_model)

        # Pass through transformer
        x = self.transformer(x, x)  # Self-attention for seq_len=1

        # Remove sequence dimension and pass through output layer
        x = x.squeeze(1)  # (batch_size, d_model)

        x = torch.cat([x, current_updrs.unsqueeze(1), time_periods.unsqueeze(1)], dim=1)
        x = self.relu(self.combined_layer(x))

        output = self.output_layer(x)  # (batch_size, 1)

        return output.squeeze(1)  # Return as (batch_size)

# Training function
def train_model(model, train_loader, val_loader=None, num_epochs=30, learning_rate=0.001, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    run_len = len(train_loader) // 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"Epoch %{len(str(num_epochs))}d/%d " % (epoch + 1,num_epochs), end="", flush=True)

        for i, (expression_data, current_updrs, time_periods, next_updrs) in enumerate(train_loader,0):
            expression_data = expression_data.to(device)
            current_updrs = current_updrs.to(device)
            time_periods = time_periods.to(device)
            next_updrs = next_updrs.to(device)

            # Forward pass
            outputs = model(expression_data, current_updrs, time_periods)
            loss = criterion(outputs, next_updrs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % run_len  == 0:
                print("#", end="", flush=True)
        print(f" Loss: {epoch_loss / len(train_loader):>.4f}", end="", flush=True)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for expression_data, current_updrs, time_periods, next_updrs in val_loader:
                    expression_data = expression_data.to(device)
                    current_updrs = current_updrs.to(device)
                    time_periods = time_periods.to(device)
                    next_updrs = next_updrs.to(device)

                    outputs = model(expression_data, current_updrs, time_periods)
                    loss = criterion(outputs, next_updrs)
                    val_loss += loss.item()
            print(f", Val Loss: {val_loss / len(val_loader):>.4f}", end="", flush=True)
        print()

    print("Training completed.")

## Test the model
def test_model(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0

    print("Testing...")

    run_len = len(dataloader) // 10

    with torch.no_grad():
        for i, (expression_data, current_updrs, time_periods, next_updrs) in enumerate(dataloader,0):
            expression_data = expression_data.to(device)
            current_updrs = current_updrs.to(device)
            time_periods = time_periods.to(device)
            next_updrs = next_updrs.to(device)

            outputs = model(expression_data, current_updrs, time_periods)
            loss = criterion(outputs, next_updrs)
            total_loss += loss.item()

            if i % run_len == 0:
                print("#", end="", flush=True)
    print(f" Test Loss: {total_loss / len(dataloader):>.4f}")
    print("Test completed.")


if __name__ == "__main__":
    df = pd.read_csv("../results/training_testing/PPMI_data_current_next.csv", index_col=0)
    ## remove rows if THERE IS MISSING DATA
    df = df.dropna(axis=0)
    num_genes = 874

    test_df = df.sample(frac=0.2)
    train_df = df.drop(test_df.index)

    train_dataset = UPDRSDataset(train_df.loc[:, train_df.columns.str.startswith("ENSG")].values, train_df["current_updrs"].values, train_df["time_period"].values, train_df["next_updrs"].values)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = UPDRSDataset(test_df.loc[:, test_df.columns.str.startswith("ENSG")].values, test_df["current_updrs"].values, test_df["time_period"].values, test_df["next_updrs"].values)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UPDRSTransformer(num_genes=num_genes)
    train_model(model, train_dataloader, val_dataloader, num_epochs=100, learning_rate=1e-4, device=device)

    torch.save(model.state_dict(), "../models/PPMI_model_B.pt")
    print("Model saved.")

