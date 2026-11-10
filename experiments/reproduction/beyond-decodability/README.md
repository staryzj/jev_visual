# Beyond Decodability reproduction records

This snapshot covers the eighth- and ninth-round Visual-JEV diagnostics.
The preceding folders contain dependencies for reproducing the reported statistics.
See the README in each round for its required scripts and data.

Use Python 3.12 and NumPy for numerical summaries and Matplotlib for figures.
Run each command from its round directory. To reproduce an archived result
without overwriting it, first rename its included JSON summary.

- Round 8: run `analyze.py` and `photo_diagnostics.py`.
- Round 9: run `selection.py`, `geometry_selection.py`, `arrow_analysis.py`,
  `arrow_effects.py`, and `prospective_analysis.py`.

The snapshot contains analysis code, fixed scores, manifests, and synthetic
scenes. It reproduces the archived statistics without model weights. Original
GPU extraction and fitting scripts record the protocol and require the
corresponding checkpoints, projection banks and runtime. Natural benchmark
photographs, manuscript sources, and model weights are excluded.
