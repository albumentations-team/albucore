# Linux dtype conversion follow-up — 2026-11-03

Linux x86-64 Tensor uint8→float16 conversion uses a float32 workspace. The initial
native half path was slower than the former float32-then-half chain on the tested
Xeon. NumPy keeps its float16 LUT; macOS ARM64 keeps its measured half Tensor path.

The [raw archive](convert_dtype_linux_20261103.json.gz) contains the initial
432-case Linux comparison, 54 initial candidate comparisons, 144 corrected public
comparisons, 54 corrected candidate comparisons, and 144 macOS control comparisons.
It preserves full package inventories, imported paths, CPU/ISA metadata, strides,
thread settings, correctness summaries and the corrected conversion source hash.

The baseline is `432f4853d5412c56e9e15f2fa01dc208b7b5e5a8`; the initial candidate is
`6300a860c48e31d21bf16a23215832335b220680`. Corrected samples use that checkout with
the Linux float32-workspace change, identified by its exact `convert.py` SHA-256.

## Results

Ratios are current / baseline runtime across the same dtype pair and output mode;
values below one are faster. Each phase interleaved its baseline and current calls
on its own VM. The before and after phases used separate VMs of the same type.

| Linux public call | Initial median | Initial max | Corrected median | Corrected max |
| --- | ---: | ---: | ---: | ---: |
| NumPy uint8 → float16 | 0.448 | 0.613 | 0.433 | 0.607 |
| Tensor uint8 → float16 | 1.756 | 2.884 | 0.993 | 1.032 |
| NumPy uint8 → float32 control | 1.004 | 1.035 | 1.002 | 1.067 |
| Tensor uint8 → float32 control | 1.004 | 1.024 | 1.004 | 1.029 |

The uint8→float32 control path keeps its existing kernel. The table retains its
slowest cell rather than treating a near-one median as a bound on every workload.

In corrected candidate measurements, direct half destination division took
1.877/1.883/1.843× public time for allocating/contiguous/strided output. A half
workspace took 1.892/2.226/1.512×. The selected float32 workspace candidate took
0.969/0.977/0.986×, with its small advantage over the public router reflecting
dispatch overhead. All three representative shapes favored float32 normalization.

The float32 workspace restores the former chain's working precision and live-buffer
arrangement: one float32 working image and a half result or supplied half output.
It uses more working memory than the initial native-half proposal. It does not add
another image buffer beyond the former two-call baseline.

The macOS recheck retained median ratios of 0.818 for uint8→float16 and 1.002 for
uint8→float32 across 144 NumPy/Tensor calls. The Linux-specific branch therefore
keeps the ARM64 kernel selection; the archive records all control cells.

## Environment and validation

Both Linux phases used temporary GCP `n2-standard-2` VMs in `us-central1-a`, with
two vCPUs and 8 GiB RAM. `lscpu` reported Intel Xeon CPU @ 2.80 GHz; NumKong reported
`haswell` and `skylake` capabilities. Processes were pinned to CPU 0. Torch intra-op
and inter-op threads were one; OpenCV parallelism was disabled, and BLAS/OpenMP
thread variables were set to one before imports.

The profile was Python 3.11.15, Torch 2.15.0+cpu, NumPy 2.4.6, OpenCV 5.0.0.93,
NumKong 7.8.7 and StringZilla 5.3.2. Workloads used HWC/CHW, the four canonical
non-square sizes and C=1/3/9, allocating/contiguous/last-axis-step-2 outputs, scale
255, ten warmup pairs and 51 measured pairs. Comparison medians use nanoseconds;
candidate raw samples use milliseconds. RNG seeds were 31 and 73 respectively.

The initial 764 and corrected 773 targeted tests passed. Every comparison checked
exact output against the previous implementation; candidates also checked input
preservation, destination identity and all 256 uint8 values at scales 255/128/17.3.
The corrected rerun covers only uint8-source conversions affected by the change;
the initial archive retains the unchanged floating-source measurements.

Results were uploaded before both temporary VMs automatically deleted themselves.
