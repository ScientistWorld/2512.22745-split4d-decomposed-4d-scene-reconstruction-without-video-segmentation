"""Paper ablation "w/o streaming learning" (Table 2).

Identical Split4D objective, representation, iteration budget, and inference;
only the observation order changes from temporally ordered streaming to
uniformly random sampling (run with ``--no-streaming``).
"""

from __future__ import annotations

from method.split4d import main as split4d_main


if __name__ == "__main__":
    split4d_main()
