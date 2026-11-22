## Main results (5-fold leave-subjects-out, MotionSense)

| Model | Params | Accuracy (%) | Macro-F1 (%) | CPU latency, 1 window (ms) |
|---|---:|---:|---:|---:|
| cnn | 96k | 95.0 ± 2.1 | 93.9 ± 2.6 | 0.17 |
| rf | - | 94.8 ± 1.4 | 93.6 ± 1.5 | 21.23 |
| tcn | 134k | 94.0 ± 1.6 | 92.7 ± 2.0 | 0.50 |
| convgru | 71k | 93.6 ± 2.3 | 92.3 ± 2.2 | 0.37 |
| transformer | 110k | 93.0 ± 2.1 | 91.3 ± 3.1 | 0.19 |

Per-class F1 (cnn): downstairs 85.7, upstairs 89.6, walking 93.6, jogging 98.2, sitting 98.3, standing 98.1

## Random-window split vs subject split

| Model | Random split F1 (%) | Subject split F1 (%) | Gap (pts) |
|---|---:|---:|---:|
| rf | 97.6 ± 0.3 | 93.6 ± 1.5 | 4.0 |
| cnn | 98.8 ± 0.2 | 93.9 ± 2.6 | 4.9 |

## Window length (1D-CNN)

| Window | Seconds | Macro-F1 (%) |
|---:|---:|---:|
| 32 | 0.64 | 90.4 ± 1.7 |
| 64 | 1.28 | 92.5 ± 2.1 |
| 128 | 2.56 | 93.9 ± 2.6 |
| 256 | 5.12 | 94.8 ± 2.5 |

## Ablations (1D-CNN, 5-fold leave-subjects-out)

| Setting | Macro-F1 (%) |
|---|---:|
| acc only | 84.2 ± 6.2 |
| acc + gyro | 88.9 ± 0.8 |
| acc + gyro + gravity | 93.9 ± 2.6 |
| acc + gyro + gravity + attitude | 91.1 ± 2.9 |
| no augmentation | 91.1 ± 3.1 |
| jitter + scale, no rotation | 89.9 ± 3.3 |
| jitter + scale + rotation (default) | 93.9 ± 2.6 |

## Streaming (decision every 0.5 s, causal smoothing)

| Model | Smoother | Accuracy (%) | Macro-F1 (%) | Label switches / min | Median delay after a change (s) |
|---|---|---:|---:|---:|---:|
| cnn | none | 93.8 | 92.4 | 2.91 | 1.4 |
| cnn | ema | 93.6 | 92.3 | 1.38 | 2.1 |
| cnn | vote | 93.2 | 91.7 | 1.60 | 2.3 |
| cnn | hmm | 93.6 | 92.2 | 1.06 | 2.2 |
| convgru | none | 92.7 | 91.1 | 3.51 | 1.1 |
| convgru | ema | 92.6 | 91.1 | 1.71 | 1.9 |
| convgru | vote | 92.3 | 90.7 | 1.80 | 2.0 |
| convgru | hmm | 92.7 | 91.1 | 1.45 | 1.8 |
| tcn | none | 92.6 | 91.1 | 2.74 | 1.5 |
| tcn | ema | 92.5 | 91.1 | 1.63 | 2.3 |
| tcn | vote | 92.0 | 90.4 | 1.78 | 2.5 |
| tcn | hmm | 92.7 | 91.5 | 1.26 | 2.3 |
| transformer | none | 92.0 | 90.0 | 5.79 | 1.5 |
| transformer | ema | 92.6 | 90.9 | 2.40 | 2.2 |
| transformer | vote | 92.2 | 90.4 | 2.41 | 2.4 |
| transformer | hmm | 92.6 | 91.0 | 2.03 | 2.1 |

(ground truth has 0.63 activity changes per minute in these streams)

Per-subject stream accuracy (cnn + hmm): min 75.6 (subject 19), max 97.8, median 96.4

