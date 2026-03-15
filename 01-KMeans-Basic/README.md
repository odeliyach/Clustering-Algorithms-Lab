# Lloyd's K-Means Clustering - Basic Implementation (v1)

## Overview

This is the **foundational implementation** of Lloyd's k-means clustering algorithm in both C and Python. It demonstrates the core algorithm without advanced optimizations.

### Core Algorithm: Lloyd's K-Means

Lloyd's algorithm is a simple yet effective clustering method that iteratively:
1. Assigns each point to the nearest centroid
2. Updates centroids as the mean of assigned points
3. Repeats until convergence

## Mathematical Foundation

### Euclidean Distance
$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{d} (p_i - q_i)^2}$$

### Lloyd's Algorithm Steps

**Initialization**: Initialize centroids as the first K datapoints:
$$\boldsymbol{\mu}_k = \mathbf{x}_k \quad \text{for } k = 1, 2, \ldots, K$$

**Assignment Step**: Assign each point to nearest centroid:
$$C_k^{(t)} = \{ \mathbf{x}_i : d(\mathbf{x}_i, \boldsymbol{\mu}_k^{(t)}) < d(\mathbf{x}_i, \boldsymbol{\mu}_j^{(t)}) \text{ for all } j \neq k \}$$

**Update Step**: Recompute centroids:
$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{\mathbf{x}_i \in C_k^{(t)}} \mathbf{x}_i$$

**Convergence Criterion**:
$$\max_k d(\boldsymbol{\mu}_k^{(t)}, \boldsymbol{\mu}_k^{(t+1)}) < \epsilon \quad (\epsilon = 0.001)$$

OR iteration count reaches maximum (default: 400)

## Implementation Details

### C Version (`lloyd_clustering.c`)
- **Pure ANSI C** with strict compliance flags
- Manual memory management
- Command-line interface
- Reads CSV data from stdin
- Output: Final centroids with 4 decimal precision

**Compilation:**
```bash
gcc -ansi -Wall -Wextra -Werror -pedantic-errors -o lloyd lloyd_clustering.c -lm
```

### Python Version (`lloyd_clustering.py`)
- Pure Python implementation
- NumPy-free (uses only standard library)
- Equivalent functionality to C version
- Easy to read and understand

**Execution:**
```bash
python3 lloyd_clustering.py <K> [max_iterations] < data.csv
```

## Usage

### Command-Line Arguments

```
./lloyd <K> [max_iterations]
python3 lloyd_clustering.py <K> [max_iterations]
```

**Parameters:**
- `K` (required): Number of clusters (integer, 1 < K < N)
- `max_iterations` (optional): Maximum iterations (integer, default: 400, range: 2-999)

### Input Format

CSV format with comma-separated numeric values:
```
1.5,2.3,4.1
2.1,2.8,3.9
1.8,2.5,4.0
```

### Output Format

Final centroids in CSV format with 4 decimal places:
```
1.7333,2.5333,4.0000
2.0500,2.6500,3.9500
```

### Example

```bash
# Generate test data
python3 << 'EOF'
import random
random.seed(42)
for _ in range(100):
    x = random.gauss(0, 1)
    y = random.gauss(0, 1)
    z = random.gauss(0, 1)
    print(f'{x:.2f},{y:.2f},{z:.2f}')
EOF
> test_data.csv

# Run clustering
./lloyd 3 300 < test_data.csv
# or
python3 lloyd_clustering.py 3 300 < test_data.csv
```

## Error Handling

Both implementations validate:
- ✓ K must be an integer > 1
- ✓ K must be < N (number of datapoints)
- ✓ max_iterations must be an integer in range [2, 999]
- ✓ Input data must be numeric CSV format
- ✓ All datapoints must have the same dimensionality

## Time Complexity

- **Per iteration**: O(N · K · d)
  - N datapoints × K distance calculations × d dimensions
- **Total**: O(I · N · K · d) where I is iterations needed

## Space Complexity

- O(N · d + K · d) for storing datapoints and centroids

## Key Learnings from v1

This basic implementation teaches:
- ✅ Core clustering algorithm logic
- ✅ Euclidean distance calculation
- ✅ ANSI C compliance and strict compilation
- ✅ Python equivalent implementations
- ✅ Input/output handling
- ✅ Memory management in C
- ✅ Convergence criteria implementation

## Limitations

- **Naive initialization**: Uses first K points (suboptimal)
- **Pure implementations**: No performance optimization
- **Single language**: Must compile/run separately
- **No K selection**: User must choose K manually

## What's Next (v2)?

See the parent directory for **v2-C-Extension-Optimized** which improves upon v1:
- K-Means++ smart initialization (3-5x faster convergence)
- Python-C hybrid architecture (5x performance boost)
- Elbow method for automatic K selection
- Professional library design
- Production-ready features

---

**Status**: Learning Foundation ✓  
**Next**: Advanced Optimization 📈
