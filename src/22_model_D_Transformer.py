import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

# Define the dataset class
class UPDRSDataset(Dataset):
    def __init__(self, genes_current, updrs_current, genes_previous, updrs_previous, updrs_next):
        self.genes_current = genes_current
        self.updrs_current = updrs_current
        self.genes_previous = genes_previous
        self.updrs_previous = updrs_previous
        self.updrs_next = updrs_next

    def __len__(self):
        return len(self.genes_current)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.genes_current[idx], dtype=torch.float32),
            torch.tensor(self.updrs_current[idx], dtype=torch.float32),
            torch.tensor(self.genes_previous[idx], dtype=torch.float32),
            torch.tensor(self.updrs_previous[idx], dtype=torch.float32),
            torch.tensor(self.updrs_next[idx], dtype=torch.float32),
        )

# Define the Transformer model
class TransformerPredictor(nn.Module):
    def __init__(self, num_genes, hidden_size=128, num_heads=8, num_layers=12, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        # Transformer encoder layers
        gene_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout)
        self.gene_transformer = nn.TransformerEncoder(gene_encoder_layer, num_layers=num_layers)

        # Linear layers for input transformation
        self.input_gene = nn.Linear(num_genes, hidden_size)
        self.input_updrs = nn.Linear(1, 1)

        # Output layer
        self.fc = nn.Linear(hidden_size + 1, 1)
        self.relu = nn.ReLU()

    def forward(self, genes_current, updrs_current, genes_previous, updrs_previous):
        # Transform gene inputs
        gene_previous = self.input_gene(genes_previous).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        gene_current = self.input_gene(genes_current).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)

        # Process gene inputs through the transformer
        genes_combined = torch.cat([gene_previous, gene_current], dim=1)  # Shape: (batch_size, seq_len=2, hidden_size)
        genes_output = self.gene_transformer(genes_combined).mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Transform UPDRS inputs
        updrs_previous = self.input_updrs(updrs_previous.unsqueeze(1)).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        updrs_current = self.input_updrs(updrs_current.unsqueeze(1)).unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        updrs_combined = torch.cat([updrs_previous, updrs_current],dim=1)  # Shape: (batch_size, seq_len=2, hidden_size)

        # Aggregate UPDRS features
        updrs_output = updrs_combined.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Combine features
        combined = torch.cat([genes_output, updrs_output], dim=1)  # Shape: (batch_size, hidden_size * 2)
        output = self.fc(combined)  # Shape: (batch_size, 1)
        return self.relu(output)


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
        for i,(genes_current, updrs_current, genes_previous, updrs_previous, updrs_next) in enumerate(train_loader,0):
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

        # Validation step
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_targets = []
            val_predictions = []
            with torch.no_grad():
                for genes_current, updrs_current, genes_previous, updrs_previous, updrs_next in val_loader:
                    genes_current, updrs_current = genes_current.to(device), updrs_current.to(device)
                    genes_previous, updrs_previous = genes_previous.to(device), updrs_previous.to(device)
                    updrs_next = updrs_next.to(device)

                    outputs = model(genes_current, updrs_current, genes_previous, updrs_previous)
                    loss = criterion(outputs.squeeze(), updrs_next)
                    val_loss += loss.item()

                    val_targets.extend(updrs_next.cpu().numpy())
                    val_predictions.extend(outputs.squeeze().cpu().numpy())

            val_r2 = r2_score(val_targets, val_predictions)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Train R²: {train_r2:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val R²: {val_r2:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Train R²: {train_r2:.4f}")

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
    dataset = UPDRSDataset(gene_expression_t2, updrs_t2, gene_expression_t1, updrs_t1, updrs_t3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(num_genes)
    train_model(model, dataloader, num_epochs=100, learning_rate=1e-3, device=device)

    # Save the model
    torch.save(model.state_dict(), 'model_D_Transformer_model.pt')