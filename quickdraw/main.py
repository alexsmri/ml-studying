import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import random

WORKING_DIR = Path(__file__).parent
BATCH_SIZE = 5

def get_imp(t: str) -> torch.Tensor:
    t_data: np.ndarray = np.load(Path(WORKING_DIR / f"data/{t}.npy"))
    t_img = torch.tensor(t_data)
    return t_img
    

def build_batch(dataset: torch.Tensor):
    didx = random.sample(range(0, dataset.shape[0]), BATCH_SIZE)
    return dataset[didx]

airplanes = build_batch(get_imp("airplane"))
airplanes = airplanes.reshape((5, 28, 28))

plt.imshow(airplanes[0], "gray")
plt.show()

NEURONS_1 = 2000
NEURONS_2 = 2000
OUTPUT = 10
W1 = torch.randn(2000, 784)
B1 = torch.randn(2000, 784)
