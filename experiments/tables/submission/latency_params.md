# Latency/Params

Parameter and timing values from recorded result.json/provenance only. Different timing boundaries are disclosed and must not be pooled into a speed comparison; controls and warm-cache policies are named explicitly. Recorded row timings exclude subsequent wrong-control repair passes.

| Model | Protocol/dataset | Trainable params | Measured mean ms | Timing boundary |
| --- | --- | --- | --- | --- |
| Visual-JEV V3 | controlled checkpoint | 4512129 | -- | parameter count from recorded provenance; timing pending common live harness |
| Visual-JEV fixed checkpoint | scienceqa | 4512129 | 165.58 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | aokvqa | 4512129 | 210.71 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | iconqa | 4512129 | 164.35 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | ai2d | 4512129 | 180.27 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | vsr | 4512129 | 182.29 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
| Visual-JEV fixed checkpoint | sugarcrepe_pp | 4512129 | 189.71 | full decision; includes control passes when enabled; repeated-image visual cache disclosed; not a controlled latency benchmark |
