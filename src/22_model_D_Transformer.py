import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, genes_current, updrs_current, genes_previous, updrs_previous, updrs_next):
        self.genes_previous = genes_previous
        self.updrs_previous = updrs_previous
        self.genes_current = genes_current
        self.updrs_current = updrs_current
        self.updrs_next = updrs_next

    def __len__(self):
        return len(self.genes_current)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.genes_previous[idx], dtype=torch.float32),
            torch.tensor(self.updrs_previous[idx], dtype=torch.float32),
            torch.tensor(self.genes_current[idx], dtype=torch.float32),
            torch.tensor(self.updrs_current[idx], dtype=torch.float32),
            torch.tensor(self.updrs_next[idx], dtype=torch.float32),
        )

# Define the Transformer model
class TransformerPredictor(nn.Module):
    def __init__(self, num_genes, hidden_size=128, num_heads=4, num_layers=6, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        # Transformer encoder layers
        gene_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.gene_transformer = nn.TransformerEncoder(gene_encoder_layer, num_layers=num_layers)

        # Linear layers for input transformation
        self.input_gene = nn.Linear(num_genes, hidden_size)
        self.input_updrs = nn.Linear(1, 1)

        # Output layer
        self.fc = nn.Linear(hidden_size + 1, 1)
        self.relu = nn.ReLU()

    def forward(self, genes_previous, updrs_previous,genes_current, updrs_current):
        # Transform gene inputs
        genes_previous = self.input_gene(genes_previous).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        genes_current = self.input_gene(genes_current).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        genes_combined = torch.cat([genes_previous, genes_current], dim=1)  # Shape: (batch_size, seq_len=2, hidden_size)
        genes_output = self.gene_transformer(genes_combined).mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Transform UPDRS inputs
        updrs_previous = self.input_updrs(updrs_previous.unsqueeze(1)).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        updrs_current = self.input_updrs(updrs_current.unsqueeze(1)).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        updrs_combined = torch.cat([updrs_previous, updrs_current],dim=1)  # Shape: (batch_size, seq_len=2, hidden_size)
        updrs_output = updrs_combined.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Combine features
        combined = torch.cat([genes_output, updrs_output], dim=1)  # Shape: (batch_size, hidden_size + 1)
        output = self.relu(self.fc(combined))  # Shape: (batch_size, 1)
        return output


# Training function
def train_model(model, train_loader, val_loader=None, num_epochs=30, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    run_len = len(train_loader) // 10

    for epoch in range(num_epochs):
        model.train()
        all_targets = []
        all_predictions = []
        epoch_loss = 0.0
        for i,(genes_previous, updrs_previous,genes_current, updrs_current, updrs_next) in enumerate(train_loader,0):
            genes_previous, updrs_previous = genes_previous.to(device), updrs_previous.to(device)
            genes_current, updrs_current = genes_current.to(device), updrs_current.to(device)
            updrs_next = updrs_next.to(device)

            # Forward pass
            outputs = model(genes_previous, updrs_previous, genes_current, updrs_current)
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
                for genes_previous, updrs_previous, genes_current, updrs_current, updrs_next in val_loader:
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


## Test the model
def test_model(model, test_dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0

    print("Testing...")

    run_len = len(test_dataloader) // 10 + 1

    test_results_df = pd.DataFrame(columns=["Actual", "Predicted"])
    with torch.no_grad():
        for i,(genes_previous, updrs_previous, genes_current, updrs_current, updrs_next) in enumerate(test_dataloader,0):
            genes_previous, updrs_previous = genes_previous.to(device), updrs_previous.to(device)
            genes_current, updrs_current = genes_current.to(device), updrs_current.to(device)
            updrs_next = updrs_next.to(device)

            outputs = model(genes_current, updrs_current, genes_previous, updrs_previous)
            loss = criterion(outputs, updrs_next)
            total_loss += loss.item()

            test_results_df = pd.concat([test_results_df, pd.DataFrame(
                {"Actual": updrs_next.cpu().numpy(), "Predicted": outputs.squeeze().cpu().numpy()})], ignore_index=True)

            if i % run_len == 0:
                print("#", end="", flush=True)

    print("")

    print(f"Test Loss: {total_loss / len(test_dataloader):>.4f}")
    print("Test completed.")

    return test_results_df

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv("../results/training_testing/PPMI_data_previous_current_next.csv", index_col=0)
    df = df.dropna(axis=0)

    gene_expression_t1 = df.loc[:, df.columns.str.startswith('previous_ENSG')]
    gene_expression_t2 = df.loc[:, df.columns.str.startswith('current_ENSG')]
    updrs_t1 = df['previous_updrs']
    updrs_t2 = df['current_updrs']
    updrs_t3 = df['next_updrs']

    # Create dataset and dataloader
    val_index = df.sample(frac=0.2).index
    train_index = df.drop(val_index).index

    train_dataset = UPDRSDataset(gene_expression_t1.loc[train_index,:].values, updrs_t1[train_index].values,
                                 gene_expression_t2.loc[train_index,:].values, updrs_t2[train_index].values, updrs_t3[train_index].values)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = UPDRSDataset(gene_expression_t1.loc[val_index,:].values, updrs_t1[val_index].values,
                               gene_expression_t2.loc[val_index,:].values, updrs_t2[val_index].values, updrs_t3[val_index].values)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    num_genes = 874
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TransformerPredictor(num_genes)
    train_model(model, train_dataloader, val_dataloader, num_epochs=100, learning_rate=1e-3, device=device)

    # Save the model
    torch.save(model.state_dict(), '../models/model_D_Transformer_model.pt')

    ## Test the model
    test_df = pd.read_csv("../results/training_testing/PPMI_data_previous_current_next.csv", index_col=0)
    test_df = test_df.dropna(axis=0)

    gene_expression_t1 = test_df.loc[:, test_df.columns.str.startswith('previous_ENSG')]
    gene_expression_t2 = test_df.loc[:, test_df.columns.str.startswith('current_ENSG')]
    updrs_t1 = test_df['previous_updrs']
    updrs_t2 = test_df['current_updrs']
    updrs_t3 = test_df['next_updrs']

    test_dataset = UPDRSDataset(gene_expression_t1.values, updrs_t1.values,
                               gene_expression_t2.values, updrs_t2.values, updrs_t3.values)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = TransformerPredictor(num_genes)
    model.load_state_dict(torch.load('../models/model_D_Transformer_model.pt'))
    model.to(device)

    test_results_df = test_model(model, test_dataloader, device)

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
    test_results_df.to_csv("../results/training_testing/PDBP_test_results.csv")
    print("Test results saved.")

