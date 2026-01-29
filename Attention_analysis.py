import torch
import matplotlib.pyplot as plt
from data_generation import generate_data
from sklearn.preprocessing import StandardScaler
from models import TimeSeriesTransformer

df = generate_data()
scaler = StandardScaler()
data = scaler.fit_transform(df.values)

SEQ_LEN = 48
X = [data[i:i+SEQ_LEN] for i in range(len(data)-SEQ_LEN-1)]
X = torch.tensor(X, dtype=torch.float32)

model = TimeSeriesTransformer(input_dim=5)
model.load_state_dict(torch.load("transformer.pth"))
model.eval()

sample = X[:1]
_ = model(sample)

attn_weights = model.get_attention_weights()
weights = attn_weights[0][0, 0].detach().cpu()

plt.imshow(weights, cmap="viridis")
plt.title("Attention Map (Layer 1 Head 1)")
plt.colorbar()
plt.show()
