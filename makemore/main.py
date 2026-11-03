from pathlib import Path
import torch

words = (Path(__file__).parent / "names.txt").read_text().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = [","] + list(w) + [","]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1