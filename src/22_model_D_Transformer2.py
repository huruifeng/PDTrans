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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
import numpy as np


# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, genes_previous, updrs_previous, genes_current, updrs_current, updrs_next):
        self.genes_previous = genes_previous
        self.updrs_previous = updrs_previous
        self.genes_current = genes_current
        self.updrs_current = updrs_current
        self.updrs_time_series = np.concatenate([self.updrs_previous, self.updrs_current])
        self.genes_time_series = np.concatenate([self.genes_previous, self.genes_current])
        self.updrs_next = updrs_next


    def __len__(self):
        return len(self.updrs_next)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.genes_previous[idx], dtype=torch.float32),
            torch.tensor(self.updrs_previous[idx], dtype=torch.float32),
            torch.tensor(self.genes_current[idx], dtype=torch.float32),
            torch.tensor(self.updrs_current[idx], dtype=torch.float32),
            torch.tensor(self.updrs_next[idx], dtype=torch.float32)
        )


# Define the Transformer model
class TransformerPredictor(nn.Module):
    def __init__(self, num_genes, hidden_size=512, num_heads=8, num_layers=12, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        self.gene_embedding = nn.Linear(num_genes, hidden_size)  # Embedding for gene
        self.updrs_embedding = nn.Linear(1, hidden_size)  # Embedding for UPDRS

        self.gene_transformer = nn.Transformer(
            d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dropout=dropout, batch_first=False
        )

        self.updrs_transformer = nn.Transformer(
            d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dropout=dropout, batch_first=False
        )

        self.fc = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, genes_previous, updrs_previous, genes_current, updrs_current):
        genes_previous = self.gene_embedding(genes_previous)
        updrs_previous = self.updrs_embedding(updrs_previous)
        genes_current = self.gene_embedding(genes_current)
        updrs_current = self.updrs_embedding(updrs_current)

        # Pass through the transformer
        gene_transformer_output = self.gene_transformer(
            src=torch.cat([genes_previous], dim=1),
            tgt=torch.cat([genes_current], dim=1)
        )

        updrs_transformer_output = self.updrs_transformer(
            src=torch.cat([updrs_previous], dim=1),
            tgt=torch.cat([updrs_current], dim=1)
        )

        output = torch.cat([gene_transformer_output, updrs_transformer_output], dim=1)

        output = self.relu(output)
        output = self.relu(self.fc(output))
        return output


# Training function
def train_model(model, train_loader, val_loader=None, num_epochs=30, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    run_len = len(train_loader) // 10 + 1

    for epoch in range(num_epochs):
        model.train()
        all_targets = []
        all_predictions = []
        epoch_loss = 0.0
        for i,(genes_previous, updrs_previous,genes_current, updrs_current, updrs_next) in enumerate(train_loader,0):
            genes_current, updrs_current = genes_current.to(device), updrs_current.to(device)
            genes_previous, updrs_previous = genes_previous.to(device), updrs_previous.to(device)
            updrs_next = updrs_next.to(device)

            # Forward pass
            outputs = model(genes_current, updrs_current, genes_previous, updrs_previous)
            loss = criterion(outputs.squeeze(), updrs_next)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_targets.extend(updrs_next.cpu().numpy())
            all_predictions.extend(outputs.squeeze().detach().cpu().numpy())

            if i % run_len  == 0:
                print(".", end="", flush=True)

        # Calculate R-squared for training set
        train_r2 = r2_score(all_targets, all_predictions)
        train_pcc = pearsonr(all_targets, all_predictions)[0]

        # Validation step
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_targets = []
            val_predictions = []
            with torch.no_grad():
                for genes_previous, updrs_previous,genes_current, updrs_current, updrs_next in val_loader:
                    genes_previous, updrs_previous = genes_previous.to(device), updrs_previous.to(device)
                    genes_current, updrs_current = genes_current.to(device), updrs_current.to(device)
                    updrs_next = updrs_next.to(device)

                    outputs = model(genes_current, updrs_current, genes_previous, updrs_previous)
                    loss = criterion(outputs.squeeze(), updrs_next)
                    val_loss += loss.item()

                    val_targets.extend(updrs_next.cpu().numpy())
                    val_predictions.extend(outputs.squeeze().cpu().numpy())

            val_r2 = r2_score(val_targets, val_predictions)
            val_pcc = pearsonr(val_targets, val_predictions)[0]
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Train R²: {train_r2:.4f}, PCC: {train_pcc:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val R²: {val_r2:.4f}, Val PCC: {val_pcc:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Train R²: {train_r2:.4f}, PCC: {train_pcc:.4f}")

    print("Training completed.")

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv("../results/training_testing/PPMI_data_previous_current_next.csv", index_col=0).dropna()

    num_genes = 874
    gene_expression_t1 = df.loc[:, df.columns.str.startswith('previous_ENSG')].values
    gene_expression_t2 = df.loc[:, df.columns.str.startswith('current_ENSG')].values
    updrs_t1 = df['previous_updrs'].values
    updrs_t2 = df['current_updrs'].values
    updrs_t3 = df['next_updrs'].values

    # Create dataset and dataloader
    dataset = UPDRSDataset(gene_expression_t1, updrs_t1,gene_expression_t2, updrs_t2,updrs_t3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(num_genes)
    train_model(model, dataloader, num_epochs=100, learning_rate=1e-3, device=device)

    # Save the model
    torch.save(model.state_dict(), '../models/model_D_Transformer2_model.pt')
