# Visual-JEV multi-turn image dialogue

`scripts/live_visual_jev_demo.py` supports bounded multi-turn memory while keeping
Qwen3-VL and VisualJEVV3 resident for the whole interactive session.

## What is implemented

- Memory is scoped to the current image. Choosing another image clears the old
  image's turns, preventing cross-image leakage.
- Every completed turn stores the question, candidate list, top-1 answer, and
  confidence.
- `auto` mode retrieves only relevant turns. Pronouns and references such as
  `it`, `this`, `它`, `这个`, and `前面` force inclusion of the immediately
  preceding turn.
- Retrieved history is bounded by both turn count and Qwen tokenizer token count.
- The selected history is inserted before the current question/candidate and the
  resulting prompt is encoded live by the Qwen language model. Cached
  `text_features` are never used.
- The history can optionally be persisted as JSON with `--history-file`.

This is candidate-based visual dialogue: each turn still needs two or more
candidate answers. VisualJEVV3 ranks those candidates; it does not generate an
unconstrained free-form answer.

## Live controls

Start once:

```bash
cd ~/projects/Open-Jev
.venv/bin/python scripts/live_visual_jev_demo.py --interactive
```

At the input menu:

- `2` or `n`: select a new image and ask the first question.
- `d`: continue the dialogue on the current image; only enter a new question and
  candidates.
- `h`: print stored turns.
- `c`: clear turns without unloading either model.
- `q`: end the session.

For memory that survives a restart:

```bash
.venv/bin/python scripts/live_visual_jev_demo.py \
  --interactive \
  --history-file runs/live_dialog_history.json
```

Useful ablations are `--history-mode off`, `--history-mode recent`, and the
default `--history-mode auto`. Prompt limits can be changed with
`--history-max-prompt-turns` and `--history-max-prompt-tokens`.

## Research basis and evaluation protocol

The implementation follows the same-image question/history/current-question
structure introduced by Visual Dialog, but uses selective retrieval instead of
blindly concatenating every turn. This choice is important because later Visual
Dialog work found that history is needed mainly for a subset of questions and
that unrestricted history can introduce shortcut behavior.

Recommended evaluation:

1. Compare `off`, `recent`, and `auto` with identical image/question/candidate
   sequences.
2. Report accuracy separately for coreference-dependent and independent turns.
3. Report history-use rate, selected turn count, prompt tokens, and latency.
4. Measure prediction changes after replacing a relevant prior answer, to verify
   that the model actually uses history rather than merely receiving it.
5. Use VisDial for natural visual dialogue and CLEVR-Dialog for controlled
   ten-round coreference diagnostics. Do not claim a VisDial/CLEVR-Dialog result
   without training or evaluating on that dataset.

Primary references:

- Das et al., *Visual Dialog*, CVPR 2017:
  https://openaccess.thecvf.com/content_cvpr_2017/html/Das_Visual_Dialog_CVPR_2017_paper.html
- Agarwal et al., *History for Visual Dialog: Do We Really Need It?*, ACL 2020:
  https://aclanthology.org/2020.acl-main.728/
- Kottur et al., *CLEVR-Dialog*, NAACL 2019:
  https://aclanthology.org/N19-1058/
- Maharana et al., *Reasoning Over the History of Multi-turn Text-to-Image
  Retrieval*, NLP-BT 2020: https://aclanthology.org/2020.nlpbt-1.9/
- Wu et al., *LongMemEval*, 2024: https://arxiv.org/abs/2410.10813

## Limitation

The current checkpoint was trained as a candidate ranker, not explicitly on
multi-turn VisDial/CLEVR-Dialog histories. The memory path is real and testable,
but dialogue accuracy must be established by an `off` versus `auto` evaluation
or by history-aware fine-tuning. Because stored answers are model predictions,
an incorrect early prediction can also propagate into later turns.
