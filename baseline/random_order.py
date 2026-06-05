from __future__ import annotations

import argparse

from method.split4d import main as split4d_main


if __name__ == "__main__":
    # This baseline is the paper's "w/o streaming learning" ablation: the same
    # feature objective and fixed Gaussian representation, but observations are
    # sampled in random temporal order.
    split4d_main()
