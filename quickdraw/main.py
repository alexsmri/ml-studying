import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f

WORKING_DIR = Path(__file__).parent
BATCH_SIZE = 5
STOI = {}
ITOS = {}


def get_imps():
    X, Y = [], []
    for t in STOI:
        t_data: np.ndarray = np.load(Path(WORKING_DIR / f"data/{t}.npy"))
        for img in t_data:
            X.append(img)
            Y.append(STOI[t])
    X = torch.tensor(np.array(X)).float()
    Y = torch.tensor(np.array(Y)).long()
    return X, Y

def build_dicts():
    data_dir = Path(WORKING_DIR / "data/")
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    files = [f.replace(".npy", "") for f in files]

    for idx, f in enumerate(files):
        STOI[f] = idx
        ITOS[idx] = f
    
    return


def build_batch():
    idx = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
    return Xtr[idx], Ytr[idx]


build_dicts()
Xtr, Ytr = get_imps()

NEURONS_1 = 2000
NEURONS_2 = 2000
OUTPUT_SIZE = len(STOI.keys())
INPUT_SIZE = 784
W1 = torch.randn(INPUT_SIZE, NEURONS_1)
B1 = torch.randn(NEURONS_1)
W2 = torch.randn(NEURONS_1, NEURONS_2)
B2 = torch.randn(NEURONS_2)

PARAMETERS = [W1, W2, B1, B2]

for p in PARAMETERS:
    p.requires_grad = True


def get_loss(p_logits, p_targets):
    loss = f.cross_entropy(p_logits, p_targets)
    return loss

def train(iterations: int):
    loss = 10

    lossi = []

    for i in range(iterations):

        batch, targets = build_batch()
        h1 = torch.tanh(batch @ W1 + B1)
        logits = torch.tanh(h1 @ W2 + B2)

        loss = get_loss(logits, targets)

        for p in PARAMETERS:
            p.grad = None

        loss.backward()
        
        lr = 0.01
        for p in PARAMETERS:
            p.data += -lr*p.grad # type: ignore

        lossi.append(loss.data)

    plt.plot(range(iterations), lossi)
    plt.show()
    return loss

train(2000)

