# Study and code review

A review of this repository as of commit `d55bab7`, carried out on 2026-11-10 on an Apple-silicon
Mac (macOS 27.0.1, arm64, MPS). It covers what the code does, whether it runs, where it departs
from the papers beyond what `FAITHFULNESS.md` already declares, and the bugs found.

Every number below comes from a run on that machine unless marked as read from the code only.

## 1. What the repository is

Six teaching implementations of the JEPA family. Each is one standalone file that depends only on
`torch` and `torchvision`, with a separate `*_extras.py` for figures and a tutorial that walks
through the code.

| File | Lines | Method | What is trained | Collapse prevention |
|---|---:|---|---|---|
| `ijepa.py` | 165 | I-JEPA | ViT encoder + predictor on CIFAR-10; 4 target blocks, 1 context block | EMA target encoder, stop-gradient |
| `vjepa.py` | 194 | V-JEPA | Tubelet (Conv3d) encoder + predictor on Moving MNIST; short and long tube masks | EMA target encoder, stop-gradient |
| `vjepa2.py` | 314 | V-JEPA 2 + AC | Phase 1 as V-JEPA; phase 2 trains a block-causal action/state predictor on the frozen encoder | EMA in phase 1; frozen encoder in phase 2 |
| `vjepa2_1.py` | 401 | V-JEPA 2.1 | Shared image/video encoder; loss on masked and visible tokens at 4 encoder depths | EMA target encoder, stop-gradient |
| `cjepa.py` | 162 | C-JEPA | Masked predictor over object slots; slots are a frozen embedding of oracle grid positions | Frozen targets |
| `leworldmodel.py` | 223 | LeWorldModel | Per-frame encoder + causal action-conditioned predictor, trained jointly | SIGReg regulariser, no EMA, no stop-gradient |

### Shared structure

All six follow the same shape, which makes them easy to compare:

1. A pre-norm transformer `Block` built on `nn.MultiheadAttention`, with fixed sine-cosine position
   embeddings registered as buffers.
2. An encoder that accepts an index tensor, so the context encoder only processes visible tokens.
3. A narrower predictor that concatenates projected context tokens with positioned mask tokens
   and returns the last `T` outputs.
4. A mask sampler written in plain Python that returns rectangular index lists. Block shapes are
   shared across a batch and index lists are trimmed to the batch minimum so they stack.
5. AdamW with weight decay on matrices only, and a linear EMA momentum schedule where an EMA is
   used.

The helpers are duplicated across files by design. The tube-mask sampler is identical in
`vjepa.py`, `vjepa2.py`, and `vjepa2_1.py`.

### How the methods differ

| | Predicts | From | Target comes from | Loss |
|---|---|---|---|---|
| I-JEPA | Embeddings of 4 image blocks | One context block minus the targets | EMA encoder on the full image, layer-normed | Smooth-L1 |
| V-JEPA | Embeddings of spatial tubes spanning all time steps | Remaining tubes | EMA encoder on the full clip, layer-normed | L1 |
| V-JEPA 2-AC | Next tubelet's latent map | Past latents, actions, states | Frozen phase-1 encoder | L1, teacher forcing + 2-step rollout |
| V-JEPA 2.1 | Masked tokens and visible tokens, 4 depths concatenated | Visible tokens | EMA encoder, 4 depths | L1; context term weighted by distance to the nearest masked token |
| C-JEPA | Masked objects' history and every object's future | Frame-0 anchor and unmasked objects' history | Frozen slot embeddings | MSE |
| LeWorldModel | Next frame embedding | Past embeddings and actions | The same encoder, not detached | MSE + SIGReg |

## 2. Does it run on this Mac

Yes, with Python 3.13. All six `*_extras.py` pipelines (training, then every figure) completed
on MPS without errors, one epoch each.

**Environment.** The system `python3` is 3.15, where `pip install -r requirements.txt` fails
because `torch==2.11.0` has no wheel. The pins resolve on Python 3.11 to 3.14. The working setup
is:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

This gives Python 3.13.18, torch 2.11.0, torchvision 0.26.0, with MPS available.

**Results.** Each pipeline was called as `main(epochs=1)` from a scratch directory, so the
tracked figures in `samples/` were not overwritten.

| Pipeline | Result | Wall time | Notes |
|---|---|---:|---|
| `cjepa_extras` | OK | 12 s | |
| `leworldmodel_extras` | OK | 34 s | Runs, but its eval-mode figures are affected by H1 |
| `vjepa2_extras` | OK | 67 s | Phase 1 and phase 2, one epoch each |
| `ijepa_extras` | OK | 703 s | Includes the CIFAR-10 download, t-SNE, and the linear probe (26.8% after 1 epoch) |
| `vjepa_extras` | OK | 58 s | Includes part of the Moving MNIST download |
| `vjepa2_1_extras` | OK | 218 s | |

Times are single runs and include dataset downloads where noted. No MPS fallback or unsupported
operator errors occurred. The default epoch counts (8 for I-JEPA, 3 to 5 for the others) were not
run to completion as part of this check.

## 3. Findings

Severity reflects how far a finding can mislead someone learning from the repository.

### High

**H1. LeWorldModel is broken in evaluation mode.**
`leworldmodel.py:68-82`, used by `leworldmodel_extras.py:58` and `:87`.

The encoder and predictor each end in `Linear + BatchNorm1d`. After training, the encoder's
pre-BatchNorm outputs have a variance near 2e-4, and the batch mean moves by more than one batch
standard deviation per optimiser step. The running mean therefore sits about 6.7 batch standard
deviations away from the current batch mean. Switching to `.eval()` shifts every embedding.

| Run | Train-mode one-step MSE | Eval-mode one-step MSE |
|---|---:|---:|
| 1 epoch | 0.38 | 28.5 |
| 2 epochs | 0.26 | 41.7 |
| 4 epochs | 0.23 | 79.7 |

In the 4-epoch run the eval-mode embeddings had a per-dimension absolute mean of 7.5, against
0.04 in train mode. Rollouts with and without actions were both near 80.

The extras' rollout plot and surprise gif call `.eval()`, so regenerating them from the current
code produces meaningless figures. The Gaussianity plot hides the problem because it
re-standardises the embeddings first.

Fix that was tested: with weights frozen, reset the BatchNorm running statistics, set
`momentum=None`, and pass 20 training batches through in train mode. Eval-mode MSE returned to
0.29 (train mode on the same batch: 0.26).

Related: the zero-action diagnostic at `leworldmodel.py:207-209` runs in train mode, so it updates
the predictor's running statistics with out-of-distribution passes on every step.

**H2. The committed figures and result numbers predate the current code.**
Commit `d24fa2f` (2026-05-15) rewrote five algorithm files, including 130 changed lines in
`vjepa2.py` and the addition of the BatchNorm projector in `leworldmodel.py`. No file in
`samples/` was regenerated afterwards; all of them date from 2026-05-12 or 2026-05-13. The commit
message itself records that full training runs were not tested. The tutorials' result sections,
including the 52.7% I-JEPA linear-probe figure in the README, describe earlier code.
`vjepa2_1_extras.py` writes six figures, none of which are committed.

### Medium

**M1. The V-JEPA 2-AC predictor does not use the action.**
After 2 phase-1 epochs and 1 phase-2 epoch, the teacher-forcing loss was 0.0301 with true actions
and 0.0300 with zeroed actions. The tutorial reports the same and attributes it to the toy data,
concluding that the machinery is correct. The diagnostic cannot support that conclusion. Causes
visible in the code:

- `vjepa2.py:282` passes `actions[:, 1:]`. Action `t+1` only moves the digit within the second
  frame of the target tubelet. The displacement between tubelets comes from action `t`, which
  the input latent already reveals because it spans two frames.
- `vjepa2.py:167-168` registers `act_token` and `state_token` as zero buffers, so they are
  neither learned nor non-zero.
- `vjepa2.py:287` and `vjepa2_extras.py:56-58` feed ground-truth states into every rollout step,
  including the zero-action baseline.
- `vjepa2.py:281` encodes the whole clip with bidirectional attention, so each latent is a
  function of later frames. Changing only frames 2–9 changed the tubelet-0 latents. A linear
  probe from tubelet-0 latents to the final digit position gave R² 0.44–0.47 against a baseline
  of 0.41–0.46 from the initial position alone, so the measurable leak is small and is not the
  main cause. This is not listed in `FAITHFULNESS.md`. The claim that the paper encodes frames
  independently at this stage is from memory and was not checked against `papers/vjepa2.pdf`.
- The predictor has spatial position embeddings only; there is no temporal position signal
  beyond what the causal mask implies.

**M2. C-JEPA's collision step leaves objects on the same cell.**
`cjepa.py:69-73` checks each pair once. Resetting a colliding pair to its previous cells can put
one of them under a third object that already moved there. Replaying the dataset's own seeds:

- 426 of 8,000 clips (5.3%) contain at least one frame where two digits share a cell
  (1,142 of 64,000 frames). The two slots then have identical embeddings.
- 4,580 times, two digits exchange cells in one step without a collision being detected.

The tutorial describes the collisions as the load-bearing part of the dataset.

**M3. The LeWorldModel rollout plot is mis-scaled.**
`leworldmodel_extras.py:67-71` sums per-batch errors and never divides by the number of batches.
With the defaults (64 clips, batch size 32) the plotted "MSE" is twice the true value.

**M4. Installation metadata.**

- `requires-python` was `>=3.10` with exact pins. The pins resolve only on Python 3.11–3.14:
  `scikit-learn==1.8.0` excludes 3.10 and `torch==2.11.0` has no 3.15 wheel. Fixed in this
  review (see section 4).
- There is no `LICENSE` file, although the README and `pyproject.toml` both state MIT.
- The README clone URL and the `pyproject.toml` homepage point to `keon/jepa`; this checkout's
  origin is `auletes/jepa`.

### Low

- **L1. Memory in V-JEPA 2.1.** `vjepa2_1.py:106-107` upsamples all of MNIST to 64×64 float32 at
  construction, a 0.98 GB tensor. With two spawned workers on macOS it is pickled into each one.
  Upsampling in `__getitem__` avoids this. (Size computed; worker memory not measured.)
- **L2. Untrained norm in V-JEPA 2.1.** `final_norm` (`vjepa2_1.py:153`, `:186`) receives no
  gradient because training always uses `return_hier=True`. The dense PCA figures in the extras
  use the `return_hier=False` path, so they show the last block through an identity-initialised
  norm, not the features the loss supervised.
- **L3. Seeding.** Only the mask sampler is seeded. Weight initialisation, shuffling, and
  augmentation are not, so the README's "Reproducibility" section covers versions only.
- **L4. Mask trimming.** Trimming to the batch minimum leaves I-JEPA with as few as 5 of 64
  context tokens (median 18), and leaves 21–36% of video tokens in neither context nor target.
  This is declared in `FAITHFULNESS.md`; the numbers are recorded here for reference.
- **L5. Unused work in C-JEPA.** `FrozenSlotEncoder` ignores the rendered video, which is still
  rendered for every sample.
- **L6. Repository hygiene.** 69 MB of paper PDFs are committed, and redistribution rights vary
  by paper. Eighteen files in `samples/` are not referenced by any tutorial. `vjepa2_1.py` uses a
  different formatting style from the other five files. The duplicated mask sampler has nothing
  guarding against drift between its three copies.
- **L7. Tutorial snippets.** Most snippet lines match the source exactly (V-JEPA 2: 49 of 49;
  LeWorldModel: 68 of 69). The I-JEPA, V-JEPA, and V-JEPA 2.1 tutorials paraphrase lightly, for
  example showing the `DataLoader` without `num_workers`.
- **L8. No tests or CI.**

## 4. Changes made during this review

Only what was needed to install and run on this machine:

- `pyproject.toml`: `requires-python` changed from `>=3.10` to `>=3.11,<3.15`.
- `.python-version` added (`3.13`) so `uv` selects a supported interpreter.
- `README.md`: Python range and venv command updated to match.
- `CLAUDE.md` and this file added.

No algorithm, extras, tutorial, or sample file was modified. None of the findings in section 3
other than the Python bound in M4 has been fixed.

## 5. What was verified correct

- SIGReg: quadrature knots, trapezoid weights, Gaussian window, and batch scaling
  (`leworldmodel.py:34-53`), by reading.
- The block-causal mask in `ACPredictor`: outputs at earlier time steps are unchanged (difference
  exactly 0) when later inputs change.
- I-JEPA context and target sets never overlapped in 76,800 sampled items; the `[0]` fallback at
  `ijepa.py:115` never fired.
- The tube-mask sampler never produced an empty context in 300 batches per configuration.
- AdaLN-zero initialisation, EMA updates, and stop-gradient placement, by reading.
- V-JEPA 2.1 loss and backward pass on MPS in both image and video modes, including
  `torch.cdist`.

## 6. Recommended order of work

1. Fix LeWorldModel evaluation (H1) and the rollout averaging (M3).
2. Fix the C-JEPA collision resolution (M2).
3. Rework the V-JEPA 2-AC action alignment, tokens, and rollout states (M1), then re-check the
   action gap.
4. Regenerate every figure in `samples/` and update the numbers in the tutorials (H2).
5. Add `LICENSE` and a smoke test that runs each `train()` for a few steps (M4, L8).
