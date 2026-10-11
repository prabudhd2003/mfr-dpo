"""Keep math libraries single-threaded in tests.

CARC login nodes cap how many threads one user may start. scipy's OpenBLAS otherwise starts one
thread per CPU core when it loads, the cap refuses them, and Python reports a fake
KeyboardInterrupt followed by a segmentation fault. Must run before numpy/scipy/torch are imported.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
