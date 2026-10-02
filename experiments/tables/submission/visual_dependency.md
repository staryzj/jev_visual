# Visual Dependency

Original, blank and wrong use each declared task score: native-choice accuracy, SugarCrepe++ strict ITT or TextVQA consensus. Semantic CF is counterfactual-direction accuracy on verified image pairs. Pair-both requires both directions correct; pair flip is reported alongside it. Unavailable verified pair tests remain --. Generic wrong-image controls are diagnostic.

| Protocol | Model | Dataset | N | Original (%) | Blank (%) | Wrong (%) | Pair N | Semantic CF (%) | Pair-both (%) | Pair flip (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled/matched | random_untrained_head | COCO | 128 | 12.50 | -- | -- | -- | -- | -- | -- |
| controlled/matched | meanpool_mlp_jev | COCO | 128 | 83.59 | -- | -- | -- | -- | -- | -- |
| controlled/matched | v2_candidate_aware | COCO | 128 | 88.28 | 88.28 | 88.28 | 48 | 41.67 | 0.00 | 0.00 |
| controlled/matched | v3_simple_postmerger | COCO | 128 | 88.28 | 46.88 | 85.16 | 48 | 62.50 | 43.75 | 56.25 |
| controlled/matched | v3_without_stage_a | COCO | 128 | 86.72 | 35.16 | 85.94 | 48 | 64.58 | 41.67 | 47.92 |
| controlled/matched | v3_without_stage_c | COCO | 128 | 87.50 | 87.50 | 87.50 | 48 | 62.50 | 2.08 | 2.08 |
| controlled/matched | v3_full | COCO | 128 | 88.28 | 39.84 | 82.03 | 48 | 81.25 | 56.25 | 60.42 |
| controlled/matched | v3_stage_a_only | COCO | 128 | 9.38 | 9.38 | 8.59 | 48 | 41.67 | 0.00 | 2.08 |
| custom independent | Visual-JEV | coco | 83 | 91.57 | 15.66 | 89.16 | -- | -- | -- | -- |
| custom independent | Visual-JEV | aokvqa | 390 | 47.44 | 28.72 | 44.87 | -- | -- | -- | -- |
| custom independent | Visual-JEV | scienceqa | 984 | 77.44 | 34.55 | 63.82 | -- | -- | -- | -- |
| custom independent | Visual-JEV | iconqa | 1838 | 75.73 | 33.57 | 39.23 | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed0 | controlled | 128 | 100.00 | -- | -- | 48 | 100.00 | 100.00 | 100.00 |
| matched Visual Jev | Official Visual Jev answer-SFT seed0 | aokvqa | 390 | 85.13 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed0 | scienceqa_image_only | 984 | 87.70 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed0 | iconqa_choice | 1838 | 82.97 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed0 | coco_hard_negative | 83 | 98.80 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed1 | controlled | 128 | 100.00 | -- | -- | 48 | 100.00 | 100.00 | 100.00 |
| matched Visual Jev | Official Visual Jev answer-SFT seed1 | aokvqa | 390 | 84.36 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed1 | scienceqa_image_only | 984 | 87.80 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed1 | iconqa_choice | 1838 | 84.06 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed1 | coco_hard_negative | 83 | 100.00 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed2 | controlled | 128 | 100.00 | -- | -- | 48 | 100.00 | 100.00 | 100.00 |
| matched Visual Jev | Official Visual Jev answer-SFT seed2 | aokvqa | 390 | 86.15 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed2 | scienceqa_image_only | 984 | 88.41 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed2 | iconqa_choice | 1838 | 85.36 | -- | -- | -- | -- | -- | -- |
| matched Visual Jev | Official Visual Jev answer-SFT seed2 | coco_hard_negative | 83 | 98.80 | -- | -- | -- | -- | -- | -- |
| full benchmark | same fixed checkpoint | scienceqa | 4241 | 28.13 | 30.42 | 31.57 | -- | -- | -- | -- |
| full benchmark | same fixed checkpoint | aokvqa | 1145 | 36.33 | 28.03 | 37.12 | -- | -- | -- | -- |
| full benchmark | same fixed checkpoint | iconqa | 6316 | 30.19 | 32.39 | 31.74 | -- | -- | -- | -- |
| full benchmark | same fixed checkpoint | ai2d | 3088 | 26.55 | 24.42 | 25.94 | -- | -- | -- | -- |
| full benchmark | same fixed checkpoint | vsr | 2195 | 53.53 | 53.80 | 53.30 | -- | -- | -- | -- |
| full benchmark | same fixed checkpoint | sugarcrepe_pp | 4757 | 43.77 | 30.54 | 42.30 | -- | -- | -- | -- |
