# sklearn-ms

CPU reference implementation of mean-shift clustering for the Chicago crime data,
using scikit-learn. Serves as a baseline to validate / benchmark the Ray-Shift GPU
mean-shift.

Clusters on the planar **X / Y COORDINATE** columns (units: feet), the natural
metric space for Euclidean mean-shift.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Fast reference run on a 10k random sample (default), bandwidth auto-estimated
python meanshift.py

# Full dataset with a fixed bandwidth (feet); ~1000-2000 ft ~= a few city blocks
python meanshift.py --sample all --bandwidth 1500

# Larger sample, custom output
python meanshift.py --sample 25000 --output clusters_25k.csv
```

### Options

| Flag          | Default                      | Meaning                                        |
|---------------|------------------------------|------------------------------------------------|
| `--input`     | `../data/chicago_crime.data` | Input `.data` file                             |
| `--output`    | `clusters.csv`               | Output `CASE#,X,Y,cluster_label`               |
| `--sample`    | `10000`                      | Points to cluster; `all` or `0` for full set   |
| `--bandwidth` | (estimated)                  | Mean-shift bandwidth in feet                   |
| `--quantile`  | `0.2`                        | Quantile for bandwidth estimation              |
| `--seed`      | `42`                         | RNG seed for reproducible sampling             |

## Notes

- `MeanShift` is ~O(n²) per iteration, so the full 232k-point dataset is slow.
  Use `--sample` for quick iteration; `bin_seeding=True` is enabled to help.
- Wall-clock fit time is printed so this can act as a CPU baseline.
