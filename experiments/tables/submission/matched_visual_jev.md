# Matched Visual Jev Comparison

Custom evaluation sets inherited from existing recorded runs. Separately trained Visual-JEV checkpoints and official answer-SFT adapters have different training provenance. This table does not establish superiority on official full benchmarks. Controlled and independent held-out sets are distinct. Official indices 0/1/2 load the released root/seed1/seed2 adapters with fixed evaluation seed 20260928; they do not estimate our model training uncertainty.

| Evaluation set | Split | Model | Seed/adapter | N | Accuracy (%) | Macro-F1 (%) | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COCO controlled | custom 128 | v3_full | 20260928 | 128 | 88.28 | 88.34 | 0.6697 |
| coco | custom held-out | Visual-JEV separately trained | 20260928 | 83 | 91.57 | 91.63 | 0.7584 |
| aokvqa | custom held-out | Visual-JEV separately trained | 20260928 | 390 | 47.44 | 47.02 | 1.6469 |
| scienceqa | custom held-out | Visual-JEV separately trained | 20260928 | 984 | 77.44 | 63.24 | 0.8540 |
| iconqa | custom held-out | Visual-JEV separately trained | 20260928 | 1838 | 75.73 | 77.28 | 1.2584 |
| controlled | custom held-out | Official Visual Jev answer-SFT | 0 | 128 | 100.00 | 100.00 | 0.0046 |
| aokvqa | custom held-out | Official Visual Jev answer-SFT | 0 | 390 | 85.13 | 84.91 | 0.4868 |
| scienceqa_image_only | custom held-out | Official Visual Jev answer-SFT | 0 | 984 | 87.70 | 70.84 | 0.3044 |
| iconqa_choice | custom held-out | Official Visual Jev answer-SFT | 0 | 1838 | 82.97 | 82.34 | 0.4326 |
| coco_hard_negative | custom held-out | Official Visual Jev answer-SFT | 0 | 83 | 98.80 | 98.77 | 0.0104 |
| controlled | custom held-out | Official Visual Jev answer-SFT | 1 | 128 | 100.00 | 100.00 | 0.0055 |
| aokvqa | custom held-out | Official Visual Jev answer-SFT | 1 | 390 | 84.36 | 84.17 | 0.5108 |
| scienceqa_image_only | custom held-out | Official Visual Jev answer-SFT | 1 | 984 | 87.80 | 71.02 | 0.3092 |
| iconqa_choice | custom held-out | Official Visual Jev answer-SFT | 1 | 1838 | 84.06 | 84.90 | 0.3660 |
| coco_hard_negative | custom held-out | Official Visual Jev answer-SFT | 1 | 83 | 100.00 | 100.00 | 0.0102 |
| controlled | custom held-out | Official Visual Jev answer-SFT | 2 | 128 | 100.00 | 100.00 | 0.0051 |
| aokvqa | custom held-out | Official Visual Jev answer-SFT | 2 | 390 | 86.15 | 85.99 | 0.4796 |
| scienceqa_image_only | custom held-out | Official Visual Jev answer-SFT | 2 | 984 | 88.41 | 71.45 | 0.2953 |
| iconqa_choice | custom held-out | Official Visual Jev answer-SFT | 2 | 1838 | 85.36 | 86.43 | 0.3345 |
| coco_hard_negative | custom held-out | Official Visual Jev answer-SFT | 2 | 83 | 98.80 | 98.77 | 0.0177 |
