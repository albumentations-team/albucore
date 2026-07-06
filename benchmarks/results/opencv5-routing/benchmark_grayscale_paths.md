## 1) uint8 per-channel multiply: `multiply_lut` vs `multiply_opencv` + clip

| shape | LUT ms | OpenCV ms | faster |
|-------|-------:|-----------:|--------|
| (256,256,1) | 0.0130 | 0.0720 | LUT |
| (512,512,1) | 0.0300 | 0.2548 | LUT |
| (1024,1024,1) | 0.0947 | 1.0000 | LUT |
| (256,256,3) | 0.0605 | 0.3960 | LUT |

## 2) float32 → uint8: NumPy vs `cv2.multiply` (grayscale)

| shape | NumPy `rint(img*255)` ms | cv2 on squeezed (H,W) ms | faster | max |f-np| on product |
|-------|-------------------------:|-------------------------:|--------|--------------:|
| (256,256,1) | 0.0476 | 0.0714 | NumPy | 2.55e+02 |
| (512,512,1) | 0.1820 | 0.2748 | NumPy | 2.55e+02 |
| (1024,1024,1) | 0.7029 | 1.0654 | NumPy | 2.55e+02 |

**Note:** `cv2.multiply(img_3d, 255)` on (H,W,1) float **differs** from `img * 255` (see max |f-np| column); do not use raw 3D cv2 multiply for float→uint8 scaling on grayscale.
