"""逐数组比较两份 baseline.py 的输出：usage: compare.py A.npz B.npz"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert sorted(a.files) == sorted(b.files), (set(a.files) ^ set(b.files))
differ = [name for name in sorted(a.files) if not np.array_equal(a[name], b[name])]
print("{} arrays, {} differ".format(len(a.files), len(differ)))
for name in differ:
    x, y = a[name].astype(float), b[name].astype(float)
    print("  {}: max rel diff {:.3e}".format(name, float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-300)))))
