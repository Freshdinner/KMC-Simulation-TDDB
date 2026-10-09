"""ADR-050：改动前后 dump_tables.py 的输出逐位比较（np.array_equal，NaN 视为相等；CSV 逐字节）。"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
failures = 0
for name in ["tables"] + ["traj_s{}".format(seed) for seed in (1, 2)]:
    before = np.load(os.path.join(HERE, "{}_before.npz".format(name)))
    after = np.load(os.path.join(HERE, "{}_after.npz".format(name)))
    assert sorted(before.files) == sorted(after.files)
    for key in before.files:
        same = np.array_equal(before[key], after[key], equal_nan=True)
        failures += not same
        print("{:8s} {:28s} {:12s} {}".format(name, key, str(before[key].shape), "identical" if same else "DIFFERENT"))
for seed in (1, 2):
    with open(os.path.join(HERE, "traj_s{}_before.csv".format(seed)), "rb") as a, \
            open(os.path.join(HERE, "traj_s{}_after.csv".format(seed)), "rb") as b:
        same = a.read() == b.read()
    failures += not same
    print("traj_s{}.csv {}".format(seed, "identical" if same else "DIFFERENT"))
print("all identical" if failures == 0 else "{} differences".format(failures))
