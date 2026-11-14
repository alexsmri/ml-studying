import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f

WORKING_DIR = Path(__file__).parent
BATCH_SIZE = 10
STOI = {}
ITOS = {}
SCALING_FACTOR = 255.0

DATA_SPLITS = [0.8, 0.9]


def get_imps(use_seed = False):
    seed = 42 if use_seed else None
    rng = np.random.default_rng(seed)
    splits = [([], []), ([], []), ([], [])]

    for t in STOI:
        t_data: np.ndarray = np.load(Path(WORKING_DIR / f"data/{t}.npy"))
        t_data = t_data[rng.permutation(len(t_data))]

        split_0 = int(t_data.shape[0]*DATA_SPLITS[0])
        split_1 = int(t_data.shape[0]*DATA_SPLITS[1])

        class_splits = (
            t_data[:split_0],
            t_data[split_0:split_1],
            t_data[split_1:]
        )
        
        for (X, Y), img in zip(splits, class_splits):
            X.extend(img)
            Y.extend([STOI[t]] * len(img))

    return [
        (
            torch.tensor(np.array(X)).float() / SCALING_FACTOR,
            torch.tensor(np.array(Y)).long()
        )
        for X, Y in splits
    ]

def build_dicts():
    data_dir = Path(WORKING_DIR / "data/")
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    files = [f.replace(".npy", "") for f in files]

    for idx, f in enumerate(files):
        STOI[f] = idx
        ITOS[idx] = f
    
    return


def build_batch(mode: int):
    X, Y = (Xtr, Ytr) if mode == 0 else (Xdev, Ydev)
    idx = torch.randint(0, X.shape[0], (BATCH_SIZE,)) # tpye: ignore
    return X[idx], Y[idx]


build_dicts()
(Xtr, Ytr), (Xdev, Ydev), (Xval, Yval) = get_imps(use_seed=True)

NEURONS_1 = 2000
NEURONS_2 = 2000
OUTPUT_SIZE = len(STOI.keys())
INPUT_SIZE = 784
W1 = torch.randn(INPUT_SIZE, NEURONS_1) / INPUT_SIZE**0.5
BNGAIN1 = torch.ones(NEURONS_1)
BNBETA1 = torch.zeros(NEURONS_1)
W2 = torch.randn(NEURONS_1, NEURONS_2) / NEURONS_1**0.5
BNGAIN2 = torch.ones(NEURONS_2)
BNBETA2 = torch.zeros(NEURONS_2)
W3 = torch.randn(NEURONS_2, OUTPUT_SIZE) / NEURONS_2**0.5
B3 = torch.zeros(OUTPUT_SIZE)
EPSILON = 1e-5
MOMENTUM = 0.99

running_means = [0, 0]
running_vars = [1, 1]

PARAMETERS = [W1, W2, BNGAIN1, BNGAIN2, BNBETA1, BNBETA2, W3, B3]

for p in PARAMETERS:
    p.requires_grad = True

def get_loss(p_logits, p_targets):
    loss = f.cross_entropy(p_logits, p_targets)
    return loss

@torch.no_grad()
def update_statistics(means, vars):
    for idx, (m, v) in enumerate(zip(means, vars)):
        running_means[idx] = running_means[idx]*MOMENTUM + (1-MOMENTUM)*m
        running_vars[idx] = running_vars[idx]*MOMENTUM + (1-MOMENTUM)*v

    return True


def forward(b, is_validation = False):
    h1preact = b @ W1
    mean1 = h1preact.mean(dim=0) if not is_validation else running_means[0]
    var1 = h1preact.var(dim=0, correction=0) if not is_validation else running_vars[0]
    h1 = BNGAIN1*((h1preact - mean1) / torch.sqrt(var1 + EPSILON)) + BNBETA1 # type: ignore
    h1 = h1.tanh()
    
    
    h2preact = h1 @ W2
    mean2 = h2preact.mean(dim=0) if not is_validation else running_means[1]
    var2 = h2preact.var(dim=0, correction=0) if not is_validation else running_vars[1]
    h2 = BNGAIN2*((h2preact - mean2) / torch.sqrt(var2 + EPSILON)) + BNBETA2 # type: ignore
    
    logits = torch.tanh(h2) @ W3 + B3

    if not is_validation:
        update_statistics([mean1, mean2], [var1, var2])

    return logits

def train(iterations: int):
    loss = 10

    lossi = []

    for i in range(iterations):
        print(i)
        batch, targets = build_batch(0)

        logits = forward(batch)

        loss = get_loss(logits, targets)

        for p in PARAMETERS:
            p.grad = None

        loss.backward()
        
        lr = 0.1
        for p in PARAMETERS:
            p.data += -lr*p.grad # type: ignore

        lossi.append(loss.data)

    plt.plot(range(iterations), lossi)
    plt.show()
    
    return loss

@torch.no_grad
def validate():
    total_loss = 0.0

    for start in range(0, len(Xdev), BATCH_SIZE):

        batch = Xdev[start:start+BATCH_SIZE]
        targets = Ydev[start:start+BATCH_SIZE]
        logits = forward(batch, is_validation=True)

        total_loss += f.cross_entropy(logits, targets, reduction="sum").item()

    return total_loss / len(Xdev)

train(10000)

print(validate())

# Best loss: 0.24326949555568073

