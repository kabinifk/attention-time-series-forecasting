import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from data_generation import generate_data
from models import TimeSeriesTransformer, LSTMBaseline

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return mean_absolute_error(trues, preds), np.sqrt(mean_squared_error(trues, preds))

if __name__ == "__main__":
    df = generate_data()
    scaler = StandardScaler()
    data = scaler.fit_transform(df.values)

    SEQ_LEN = 48
    X, Y = [], []
    for i in range(len(data) - SEQ_LEN - 1):
        X.append(data[i:i+SEQ_LEN])
        Y.append(data[i+SEQ_LEN])
    X, Y = np.array(X), np.array(Y)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X[-300:], dtype=torch.float32),
        torch.tensor(Y[-300:], dtype=torch.float32)
    ), batch_size=64)

    transformer = TimeSeriesTransformer(input_dim=5)
    lstm = LSTMBaseline(input_dim=5)
    transformer.load_state_dict(torch.load("transformer.pth"))
    lstm.load_state_dict(torch.load("lstm.pth"))

    print("Transformer:", evaluate(transformer, test_loader))
    print("LSTM:", evaluate(lstm, test_loader))
