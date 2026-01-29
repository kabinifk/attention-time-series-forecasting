import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from data_generation import generate_data
from models import TimeSeriesTransformer, LSTMBaseline

SEQ_LEN = 48

def create_sequences(data):
    X, Y = [], []
    for i in range(len(data) - SEQ_LEN - 1):
        X.append(data[i:i+SEQ_LEN])
        Y.append(data[i+SEQ_LEN])
    return np.array(X), np.array(Y)

def train_model(model, loader, epochs=15):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={total_loss/len(loader):.4f}")

if __name__ == "__main__":
    df = generate_data()
    scaler = StandardScaler()
    data = scaler.fit_transform(df.values)

    X, Y = create_sequences(data)
    train_size = int(0.7 * len(X))

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X[:train_size], dtype=torch.float32),
        torch.tensor(Y[:train_size], dtype=torch.float32)
    ), batch_size=64, shuffle=True)

    transformer = TimeSeriesTransformer(input_dim=5)
    lstm = LSTMBaseline(input_dim=5)

    print("Training Transformer")
    train_model(transformer, train_loader)

    print("Training LSTM")
    train_model(lstm, train_loader)

    torch.save(transformer.state_dict(), "transformer.pth")
    torch.save(lstm.state_dict(), "lstm.pth")
