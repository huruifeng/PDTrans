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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, genes_current, updrs_current, genes_previous, updrs_previous, time_period, updrs_next):
        self.genes_current = genes_current
        self.updrs_current = updrs_current
        self.genes_previous = genes_previous
        self.updrs_previous = updrs_previous
        self.time_period = time_period
        self.updrs_next = updrs_next

    def __len__(self):
        return len(self.genes_current)

    def __getitem__(self, idx):
        return (
            self.genes_current[idx],
            self.updrs_current[idx],
            self.genes_previous[idx],
            self.updrs_previous[idx],
            self.time_period[idx],
            self.updrs_next[idx],
        )


# Define the Transformer model
class TransformerPredictor(nn.Module):
    def __init__(self, num_genes, hidden_size=128, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Linear(num_genes + 1, hidden_size)  # Embedding for gene + UPDRS data

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, 1)  # Fully connected layer for regression

    def forward(self, genes_time_series, updrs_time_series):
        # Concatenate gene expression and UPDRS at each time point
        x = torch.cat([genes_time_series, updrs_time_series.unsqueeze(-1)], dim=2)

        # Pass through embedding layer
        x = self.embedding(x)  # Shape: (batch_size, seq_len, hidden_size)
        x = x.permute(1, 0, 2)  # Transformer expects input as (seq_len, batch_size, hidden_size)

        # Pass through Transformer encoder
        transformer_out = self.transformer(x)  # Shape: (seq_len, batch_size, hidden_size)

        # Use the output corresponding to the last time step
        last_out = transformer_out[-1, :, :]  # Shape: (batch_size, hidden_size)

        # Pass through fully connected layer
        output = self.fc(last_out)
        return output.squeeze(1)  # Return as (batch_size)


# Training function
def train_model(model, dataloader, num_epochs, learning_rate, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for genes_time_series, updrs_time_series, target_updrs in dataloader:
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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    num_samples = 1000
    num_genes = 874
    time_points = 3

    genes_time_series = np.random.rand(num_samples, time_points, num_genes)
    updrs_time_series = np.random.rand(num_samples, time_points)

    dataset = UPDRSDataset(genes_time_series, updrs_time_series)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerPredictor(num_genes=num_genes)
    train_model(model, dataloader, num_epochs=20, learning_rate=1e-3, device=device)

    ## Save the model
    torch.save(model.state_dict(), 'model_D_Transformer_model.pt')


