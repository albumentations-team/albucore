## 1) uint8 per-channel multiply: `multiply_lut` vs `multiply_opencv` + clip

| shape | LUT ms | OpenCV ms | faster |
|-------|-------:|-----------:|--------|
| (256,256,1) | 0.0117 | 0.0671 | LUT |
| (512,512,1) | 0.0299 | 0.3075 | LUT |
| (1024,1024,1) | 0.1039 | 1.5207 | LUT |
| (256,256,3) | 0.0626 | 0.4656 | LUT |

## 2) float32 → uint8: NumPy vs `cv2.multiply` (grayscale)

| shape | NumPy `rint(img*255)` ms | cv2 on squeezed (H,W) ms | faster | max |f-np| on product |
|-------|-------------------------:|-------------------------:|--------|--------------:|
| (256,256,1) | 0.0543 | 0.0686 | NumPy | 2.55e+02 |
| (512,512,1) | 0.2311 | 0.3175 | NumPy | 2.55e+02 |
| (1024,1024,1) | 1.0891 | 1.5906 | NumPy | 2.55e+02 |

**Note:** `cv2.multiply(img_3d, 255)` on (H,W,1) float **differs** from `img * 255` (see max |f-np| column); do not use raw 3D cv2 multiply for float→uint8 scaling on grayscale.
