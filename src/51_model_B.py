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
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

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

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, expression_data, current_updrs, time_periods, next_updrs, indices=None):
        self.expression_data = expression_data
        self.current_updrs = current_updrs
        self.time_periods = time_periods
        self.next_updrs = next_updrs
        self.indices = indices  # Add indices

    def __len__(self):
        return len(self.next_updrs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.expression_data[idx], dtype=torch.float32),
            torch.tensor(self.current_updrs[idx], dtype=torch.float32),
            torch.tensor(self.time_periods[idx], dtype=torch.float32),
            torch.tensor(self.next_updrs[idx], dtype=torch.float32),
            self.indices[idx],  # Include the index
        )

# Define the Transformer-based model
class UPDRSTransformer(nn.Module):
    def __init__(self, num_genes, d_model=512, nhead=8, num_layers=12, dropout=0.1):
        super(UPDRSTransformer, self).__init__()
        self.input_layer = nn.Linear(num_genes, d_model) # Input layer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.combined_layer = nn.Linear(d_model + 1, 16)
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
        x = self.transformer.encoder(x)  # Self-attention for seq_len=1

        # Remove sequence dimension and pass through output layer
        x = x.squeeze(1)  # (batch_size, d_model)

        # x = torch.cat([x, current_updrs.unsqueeze(1), time_periods.unsqueeze(1)], dim=1)
        x = torch.cat([x, current_updrs.unsqueeze(1)], dim=1)
        x = self.relu(self.combined_layer(x))

        output = self.output_layer(x)  # (batch_size, 1)

        return output.squeeze(1)  # Return as (batch_size)

# Training function
def train_model(model, train_loader, val_loader=None, num_epochs=30, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    run_len = len(train_loader) // 10 + 1

    loss_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Val Loss"])
    for epoch in range(num_epochs):
        model.train()
        all_targets = []
        all_predictions = []
        epoch_loss = 0.0
        print(f"Epoch %{len(str(num_epochs))}d/%d " % (epoch + 1,num_epochs), end="", flush=True)

        for i, (expression_data, current_updrs, time_periods, next_updrs, indices) in enumerate(train_loader,0):
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

            all_targets.extend(next_updrs.cpu().numpy())
            all_predictions.extend(outputs.cpu().detach().numpy())

            if i % run_len  == 0:
                print("#", end="", flush=True)
        # Calculate R² score for the epoch
        r2 = r2_score(all_targets, all_predictions)
        pcc = pearsonr(all_targets, all_predictions)[0]
        print(f" Loss: {epoch_loss / len(train_loader):>.4f}, R²: {r2:.4f}, PCC: {pcc:.4f}", end="", flush=True)

        if val_loader is not None:
            model.eval()
            all_targets = []
            all_predictions = []
            val_loss = 0.0
            with torch.no_grad():
                for expression_data, current_updrs, time_periods, next_updrs, indices in val_loader:
                    expression_data = expression_data.to(device)
                    current_updrs = current_updrs.to(device)
                    time_periods = time_periods.to(device)
                    next_updrs = next_updrs.to(device)

                    outputs = model(expression_data, current_updrs, time_periods)
                    loss = criterion(outputs, next_updrs)
                    val_loss += loss.item()

                    # Store predictions and targets for R² calculation
                    all_targets.extend(next_updrs.cpu().numpy())
                    all_predictions.extend(outputs.cpu().numpy())
            # Calculate R² score for the epoch
            r2 = r2_score(all_targets, all_predictions)
            pcc = pearsonr(all_targets, all_predictions)[0]
            print(f", Val Loss: {val_loss / len(val_loader):>.4f}, R²: {r2:.4f}, PCC: {pcc:.4f}", end="", flush=True)

        loss_df.loc[epoch] = [epoch, epoch_loss / len(train_loader), val_loss / len(val_loader) if val_loader is not None else np.nan]
        print()

    print("Training completed.")
    return loss_df

## Test the model
def test_model(model, dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    model.to(device)
    model.eval()
    total_loss = 0.0

    print("Testing...")

    run_len = len(dataloader) // 10 + 1

    results = []
    with torch.no_grad():
        for i, (expression_data, current_updrs, time_periods, next_updrs, indices) in enumerate(dataloader,0):
            expression_data = expression_data.to(device)
            current_updrs = current_updrs.to(device)
            time_periods = time_periods.to(device)
            next_updrs = next_updrs.to(device)

            outputs = model(expression_data, current_updrs, time_periods)
            loss = criterion(outputs, next_updrs)
            total_loss += loss.item()

            if i % run_len == 0:
                print("#", end="", flush=True)

            for idx, actual, predicted in zip(indices,next_updrs.cpu().detach().numpy(), outputs.cpu().detach().numpy()):
                results.append((idx, actual.item(), predicted.item()))

    print("")
    results_df = pd.DataFrame(results, columns=["Sample","Actual", "Predicted"])
    print(f"Test Loss: {total_loss / len(dataloader):>.4f}")
    print("Test completed.")

    return results_df


if __name__ == "__main__":
    data_folder = "UPDRS1"
    df = pd.read_csv(f"../results/training_testing/{data_folder}/PPMI_data_current_next.csv", index_col=0)
    ## remove rows if THERE IS MISSING DATA
    df = df.dropna(axis=0)
    df = df.loc[df["time_period"] == 12,:]
    num_genes = 874

    val_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(val_df.index)

    train_dataset = UPDRSDataset(train_df.loc[:, train_df.columns.str.startswith("ENSG")].values, train_df["current_updrs"].values, train_df["time_period"].values, train_df["next_updrs"].values, train_df.index.values)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = UPDRSDataset(val_df.loc[:, val_df.columns.str.startswith("ENSG")].values, val_df["current_updrs"].values, val_df["time_period"].values, val_df["next_updrs"].values, val_df.index.values)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    ## PCC between current and next updrs in validation set
    pcc = pearsonr(val_df["current_updrs"].values, val_df["next_updrs"].values)[0]
    r2 = r2_score(val_df["current_updrs"].values, val_df["next_updrs"].values)
    print(f"PCC between current and next updrs in validation set: {pcc:.4f}, R^2: {r2:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UPDRSTransformer(num_genes=num_genes)
    loss_df = train_model(model, train_dataloader, val_dataloader, num_epochs=100, learning_rate=1e-4, device=device)

    loss_df.to_csv(f"../results/training_testing/{data_folder}/PPMI_model_B_loss.csv")
    print("Loss saved.")

    torch.save(model.state_dict(), f"../results/training_testing/{data_folder}/model_B.pt")
    print("Model saved.")


    ## Test the model
    test_df = pd.read_csv(f"../results/training_testing/{data_folder}/PDBP_data_current_next.csv", index_col=0)
    test_df = test_df.dropna(axis=0)
    test_df = test_df.loc[test_df["time_period"] == 12, :]

    test_dataset = UPDRSDataset(test_df.loc[:, test_df.columns.str.startswith("ENSG")].values, test_df["current_updrs"].values, test_df["time_period"].values, test_df["next_updrs"].values,  test_df.index.values)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ## PCC between current and next updrs in test set
    pcc = pearsonr(test_df["current_updrs"].values, test_df["next_updrs"].values)[0]
    r2 = r2_score(test_df["current_updrs"].values, test_df["next_updrs"].values)
    print(f"PCC between current and next updrs in testing set: {pcc:.4f}, R^2: {r2:.4f}")

    model = UPDRSTransformer(num_genes=num_genes)
    model.load_state_dict(torch.load(f"../results/training_testing/{data_folder}/model_B.pt"))
    test_results_df = test_model(model, test_dataloader, device=device)

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

    ## Save the test results
    test_results_df.to_csv(f"../results/training_testing/{data_folder}/PDBP_model_B_test_results.csv")
    print("Test results saved.")




