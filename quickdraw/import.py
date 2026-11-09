import urllib.request
from pathlib import Path

categories = [
    "dog", "tree", "house", "car",
    "bicycle", "airplane", "fish", "apple", "clock"
]

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

for category in categories:
    url = (
        "https://storage.googleapis.com/"
        f"quickdraw_dataset/full/numpy_bitmap/{category}.npy"
    )

    destination = data_dir / f"{category}.npy"

    if not destination.exists():
        urllib.request.urlretrieve(url, destination)

    print(f"Downloaded: {category}")