# Custom Checkpoint Metrics

Historical controlled and independently trained custom checkpoints have distinct provenance. These held-out partitions are not official full benchmark tests and are not target-backbone transfer results.

| Evaluation set | Split | Model | Seed | N | Accuracy (%) | Macro-F1 (%) | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COCO controlled | custom 128 | v3_full | 20260928 | 128 | 88.28 | 88.34 | 0.6697 |
| coco | custom held-out | Visual-JEV separately trained | 20260928 | 83 | 91.57 | 91.63 | 0.7584 |
| aokvqa | custom held-out | Visual-JEV separately trained | 20260928 | 390 | 47.44 | 47.02 | 1.6469 |
| scienceqa | custom held-out | Visual-JEV separately trained | 20260928 | 984 | 77.44 | 63.24 | 0.8540 |
| iconqa | custom held-out | Visual-JEV separately trained | 20260928 | 1838 | 75.73 | 77.28 | 1.2584 |
