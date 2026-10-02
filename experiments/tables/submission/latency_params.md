# Latency/Params

Parameter and timing values from recorded result.json/provenance only. Different timing boundaries are disclosed and must not be pooled into a speed comparison; controls and warm-cache policies are named explicitly. Recorded row timings exclude subsequent wrong-control repair passes.

| Model | Protocol/dataset | Trainable params | Measured mean ms | Timing boundary |
| --- | --- | --- | --- | --- |
| Visual-JEV V3 | controlled checkpoint | 4512129 | -- | parameter count from recorded provenance; timing pending common live harness |
| Official Visual Jev answer-SFT seed0 | controlled | -- | 199.91 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed0 | aokvqa | -- | 197.90 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed0 | scienceqa_image_only | -- | 175.18 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed0 | iconqa_choice | -- | 144.54 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed0 | coco_hard_negative | -- | 200.00 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed1 | controlled | -- | 204.55 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed1 | aokvqa | -- | 202.31 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed1 | scienceqa_image_only | -- | 174.39 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed1 | iconqa_choice | -- | 135.44 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed1 | coco_hard_negative | -- | 160.30 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed2 | controlled | -- | 213.61 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed2 | aokvqa | -- | 156.43 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed2 | scienceqa_image_only | -- | 137.31 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed2 | iconqa_choice | -- | 105.94 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Official Visual Jev answer-SFT seed2 | coco_hard_negative | -- | 155.02 | official model-run timing; prepare_group excluded; incomparable with full live decision |
| Visual-JEV fixed checkpoint | scienceqa | 4512129 | 165.58 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | aokvqa | 4512129 | 210.71 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | iconqa | 4512129 | 164.35 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | ai2d | 4512129 | 180.27 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | vsr | 4512129 | 182.29 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | sugarcrepe_pp | 4512129 | 189.71 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
