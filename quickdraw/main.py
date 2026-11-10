import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch
import random

WORKING_DIR = Path(__file__).parent
BATCH_SIZE = 5
STOI = {}
ITOS = {}


def get_imps() -> list:
    data = []
    for t in STOI:
        t_data: np.ndarray = np.load(Path(WORKING_DIR / f"data/{t}.npy"))
        t_img = torch.tensor(t_data)
        data.append(t_img)
    return data

def build_dicts():
    data_dir = Path(WORKING_DIR / "data/")
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    files = [f.replace(".npy", "") for f in files]

    for idx, f in enumerate(files):
        STOI[f] = idx
        ITOS[idx] = f
    
    return


def build_batch():
    batch = []
    targets = []
    type_idx = random.sample(range(0, len(ITOS.keys())), BATCH_SIZE)
    
    for t in type_idx:
        data = DATA[t]
        data_idx = random.randrange(0, data.shape[0])
        batch.append(data[data_idx])
        print(data[data_idx].shape)
        targets.append(t)

    batch = torch.stack(batch)
    return batch, targets


build_dicts()
DATA: list = get_imps()

BATCH, TARGETS = build_batch()
print(BATCH.shape)

NEURONS_1 = 2000
NEURONS_2 = 2000
OUTPUT_SIZE = len(STOI.keys())
INPUT_SIZE = 784
W1 = torch.randn(INPUT_SIZE, NEURONS_1)
B1 = torch.randn(INPUT_SIZE, NEURONS_1)
W2 = torch.randn(NEURONS_1, NEURONS_2)
B2 = torch.randn(NEURONS_1, NEURONS_2)

PARAMETERS = [W1, W2, B1, B2]

# def forward(iterations: int):
#     for i in range(iterations):
#         data = build_batch(get_imp("airplane"))
#         data = data.reshape((5, 28, 28))
#         h1 = data @ W1 + B1
#         h1 = h1.tanh()
#         h2 = h1 @ W2 + B2
#         out = h2.tanh()

