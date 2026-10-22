# Synthetic I/Q modulation classification — evaluation

> **All waveforms are simulated.** No SDR, RF receiver, antenna or real
> radio measurement was used. These numbers describe this simulator only.

Test partition: **1350 feature rows** from **1350 distinct scenarios** and **1350 reserved seeds**.

## Overall test results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline (`bpsk`) | 0.3333 | 0.1667 |
| Random Forest | **0.9593** | **0.9593** |

## Macro-F1 by SNR

| SNR (dB) | Support | Random Forest | Majority baseline |
|---:|---:|---:|---:|
| -5 | 225 | **0.7600** | 0.1667 |
| +0 | 225 | **0.9956** | 0.1667 |
| +5 | 225 | **1.0000** | 0.1667 |
| +10 | 225 | **1.0000** | 0.1667 |
| +15 | 225 | **1.0000** | 0.1667 |
| +20 | 225 | **1.0000** | 0.1667 |

## Per-class recall by SNR

| SNR (dB) | BPSK | QPSK | 8-PSK |
|---:|---:|---:|---:|
| -5 | 1.000 | 0.640 | 0.640 |
| +0 | 1.000 | 1.000 | 0.987 |
| +5 | 1.000 | 1.000 | 1.000 |
| +10 | 1.000 | 1.000 | 1.000 |
| +15 | 1.000 | 1.000 | 1.000 |
| +20 | 1.000 | 1.000 | 1.000 |

## Confusion matrices

Each row is the true class and each column is the predicted class.

### SNR -5 dB

| true \ predicted | `bpsk` | `qpsk` | `8psk` |
|---|---:|---:|---:|
| `bpsk` | 75 | 0 | 0 |
| `qpsk` | 0 | 48 | 27 |
| `8psk` | 0 | 27 | 48 |

### SNR +0 dB

| true \ predicted | `bpsk` | `qpsk` | `8psk` |
|---|---:|---:|---:|
| `bpsk` | 75 | 0 | 0 |
| `qpsk` | 0 | 75 | 0 |
| `8psk` | 0 | 1 | 74 |

### SNR +5 dB

| true \ predicted | `bpsk` | `qpsk` | `8psk` |
|---|---:|---:|---:|
| `bpsk` | 75 | 0 | 0 |
| `qpsk` | 0 | 75 | 0 |
| `8psk` | 0 | 0 | 75 |

### SNR +10 dB

| true \ predicted | `bpsk` | `qpsk` | `8psk` |
|---|---:|---:|---:|
| `bpsk` | 75 | 0 | 0 |
| `qpsk` | 0 | 75 | 0 |
| `8psk` | 0 | 0 | 75 |

### SNR +15 dB

| true \ predicted | `bpsk` | `qpsk` | `8psk` |
|---|---:|---:|---:|
| `bpsk` | 75 | 0 | 0 |
| `qpsk` | 0 | 75 | 0 |
| `8psk` | 0 | 0 | 75 |

### SNR +20 dB

| true \ predicted | `bpsk` | `qpsk` | `8psk` |
|---|---:|---:|---:|
| `bpsk` | 75 | 0 | 0 |
| `qpsk` | 0 | 75 | 0 |
| `8psk` | 0 | 0 | 75 |

## Targets

- Macro-F1 ≥ 0.90 at SNR ≥ 15 dB: **True** ({'+15': 1.0, '+20': 1.0})
- Macro-F1 ≥ 0.55 at −5 dB: **True** (0.76)

These are simulator-review targets, not operational radio performance.

## Where the classes are confused

| SNR (dB) | True | Predicted | Count |
|---:|---|---|---:|
| -5 | `8psk` | `qpsk` | 27 |
| -5 | `qpsk` | `8psk` | 27 |
| +0 | `8psk` | `qpsk` | 1 |

## Representative test errors

| Scenario | SNR | True | Predicted | Confidence |
|---|---:|---|---|---:|
| `test-qpsk--5db-0000` | -5 | `qpsk` | `8psk` | 0.553 |
| `test-qpsk--5db-0002` | -5 | `qpsk` | `8psk` | 0.490 |
| `test-qpsk--5db-0010` | -5 | `qpsk` | `8psk` | 0.440 |
| `test-qpsk--5db-0011` | -5 | `qpsk` | `8psk` | 0.443 |
| `test-qpsk--5db-0014` | -5 | `qpsk` | `8psk` | 0.537 |
| `test-qpsk--5db-0015` | -5 | `qpsk` | `8psk` | 0.480 |
| `test-qpsk--5db-0016` | -5 | `qpsk` | `8psk` | 0.533 |
| `test-qpsk--5db-0020` | -5 | `qpsk` | `8psk` | 0.530 |
| `test-qpsk--5db-0025` | -5 | `qpsk` | `8psk` | 0.610 |
| `test-qpsk--5db-0028` | -5 | `qpsk` | `8psk` | 0.547 |
| `test-qpsk--5db-0029` | -5 | `qpsk` | `8psk` | 0.540 |
| `test-qpsk--5db-0030` | -5 | `qpsk` | `8psk` | 0.513 |
| `test-qpsk--5db-0031` | -5 | `qpsk` | `8psk` | 0.507 |
| `test-qpsk--5db-0038` | -5 | `qpsk` | `8psk` | 0.530 |
| `test-qpsk--5db-0039` | -5 | `qpsk` | `8psk` | 0.533 |

## Limitations

- The channel has only global phase and AWGN; it does not model
  multipath, fading, IQ imbalance, DC offset, timing error or carrier
  frequency offset.
- The classifier sees engineered features, not a synchronized
  constellation; the feature set is intentionally small.
- The test split uses reserved simulated seeds, but no hardware
  capture has been validated.
- Per-SNR results depend on the random seeds and the synthetic
  simulator; they are not operational SDR performance claims.
