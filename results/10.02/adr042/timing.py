"""新陷阱项的单次求值耗时与量级。usage: timing.py STATE"""
import sys
import time

import numpy as np

from common import load_state
from kmc_leakage.tddb import generation_cells, generation_rates

config, inner, p, _ = load_state(sys.argv[1])
cells = generation_cells(config.device, config.tddb.generation_grid)
tick = time.time()
table = generation_rates(inner, p, config.tddb, cells)
wall = time.time() - tick
traps = table.traps
src = traps.rates > 0
print("{} N={} sources={} sum_G cells={:.4e} traps={:.4e} /s  trap share={:.3e}  wall={:.1f}s  "
      "n_intervals median/max={}/{} (r_max {:.2f}/{:.2f} nm)".format(
          sys.argv[1], inner.n_defects, int(src.sum()), table.cells.sum(), traps.rates.sum(),
          traps.rates.sum() / (traps.rates.sum() + table.cells.sum()), wall,
          int(np.median(traps.n_intervals[src])), int(traps.n_intervals.max()),
          0.5 + 0.04 * np.median(traps.n_intervals[src]), 0.5 + 0.04 * traps.n_intervals.max()), flush=True)
