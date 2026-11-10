# Visual-JEV eighth-round evidence

Run analyze.py and photo_diagnostics.py from the round-8 directory in this archive using Python 3.12 and NumPy. Keep or rename the supplied JSON summaries before rerunning. The packaged scores reproduce the validation selection summaries, fixed-scene intervals and linked patch statistics without requiring checkpoints. plot_results.py regenerates the primary figure with Matplotlib.

Each encoder contributes 72 configurations, crossing two stages, three projection widths, three projection seeds and four train/validation repartitions. The same 27 development scenes supply all partitions. The archive contains all family scores, validation losses, seed accuracies, new renders, slot masks, and the paired cover image assets. It also supplies the prior photo descriptors and predictions needed for the re-pairing analysis. Benchmark photographs are referenced by dataset IDs and hashes.

prepare.py, worker.py, fit.py and fit_only.py document feature extraction and fitting. Full-size projection banks are retained with the original runs. The original worker receipts show a module-resolution error after feature extraction and frozen scoring completed. The fit receipts document successful explicit-module resumes using the completed features and unchanged fitting code.
