import torch
import torchvision.transforms.functional as F
import numpy as np
import time
import pytest

@pytest.mark.parametrize("dtype", [np.uint8])
def test_hflip_torchvision_benchmark(dtype):
    img = np.random.randint(0, 256, (256, 256, 3), dtype=dtype)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    
    start = time.time()
    flipped = F.hflip(img_tensor)
    end = time.time()
    
    print(f"torchvision hflip time: {end - start:.6f}s")
