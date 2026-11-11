# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

Minimal, single-file PyTorch teaching implementations of the JEPA family, each with a paired
tutorial. See `README.md` for the overview, `FAITHFULNESS.md` for what each file preserves from
its paper, and `STUDY.md` for a code review with known bugs and verification results.

| Algorithm file | Extras (plots) | Tutorial | Data |
|---|---|---|---|
| `ijepa.py` | `ijepa_extras.py` | `ijepa_tutorial.md` | CIFAR-10 |
| `vjepa.py` | `vjepa_extras.py` | `vjepa_tutorial.md` | Moving MNIST |
| `vjepa2.py` | `vjepa2_extras.py` | `vjepa2_tutorial.md` | MNIST (synthetic clips) |
| `vjepa2_1.py` | `vjepa2_1_extras.py` | `vjepa2_1_tutorial.md` | MNIST + Moving MNIST |
| `cjepa.py` | `cjepa_extras.py` | `cjepa_tutorial.md` | MNIST (synthetic clips) |
| `leworldmodel.py` | `leworldmodel_extras.py` | `leworldmodel_tutorial.md` | MNIST (synthetic clips) |

## Environment

The dependency pins only resolve on Python 3.11–3.14. The system `python3` on the development
Mac is 3.15, where `torch==2.11.0` has no wheel, so always use the project venv:

```bash
uv venv --python 3.13 .venv          # .python-version pins 3.13
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python cjepa.py            # fastest end-to-end check (~10 s per epoch on MPS)
```

- Device is auto-selected: CUDA, then MPS, then CPU (`pick_device()` in every file).
- Datasets download to `./data/` on first run (git-ignored). Moving MNIST is ~780 MB.
- There is no test suite, linter config, or CI.

## Running

```bash
.venv/bin/python <algo>.py           # train only, prints losses
.venv/bin/python <algo>_extras.py    # train, then write figures to ./samples/
```

`*_extras.py` overwrites the tracked figures in `samples/`. To smoke-test without touching them,
run from another directory (figures and `./data` are resolved relative to the working directory):

```bash
mkdir -p /tmp/jepa-run && cd /tmp/jepa-run && ln -s <repo>/data data
<repo>/.venv/bin/python -c "import sys; sys.path.insert(0, '<repo>'); import cjepa_extras as m; m.main(epochs=1)"
```

Every `train()` / `main()` takes `epochs` (and `device`) keyword arguments; use `epochs=1` for
quick checks. All data loaders use `num_workers=2`, and macOS spawns workers, so any script that
calls `train()` must do so under `if __name__ == "__main__":`.

## Conventions

- **Each algorithm file is standalone.** It imports only `torch` and `torchvision`. Helpers
  (`Block`, `sincos_*`, `param_groups`, `ema_update`, `pick_device`, the tube-mask sampler) are
  copy-pasted on purpose. Do not factor them into a shared module. When changing a duplicated
  helper, apply the same change to every copy.
- **Extras stay separate.** Plotting, probes, and anything needing matplotlib or scikit-learn go
  in `<algo>_extras.py`, never in the algorithm file.
- **Keep the files small and dense.** `ijepa.py`, `vjepa.py`, `vjepa2.py`, `cjepa.py`, and
  `leworldmodel.py` use a compact style (several statements per line, inline equation comments).
  `vjepa2_1.py` uses a conventional formatted style. Match the file you are editing.
- **Docs move with code.** A change to an algorithm file needs matching updates to:
  the tutorial's code snippets and hyperparameter list, the line count in the `README.md` table
  and the tutorial title, and `FAITHFULNESS.md` if a load-bearing idea or a simplification changes.
- **Figures go stale silently.** If training behaviour changes, regenerate the affected
  `samples/` figures with the extras script and re-check the numbers quoted in the tutorial.
- Only the mask sampler is seeded (`random.Random(0)`); runs are not bit-reproducible.

## Known issues

`STUDY.md` lists them with evidence. The ones most likely to mislead while working here:

- `leworldmodel.py`: BatchNorm running statistics lag the batch statistics, so `.eval()` shifts
  the embeddings and evaluation-mode errors are 100× the training loss. The extras' rollout plot
  and surprise gif call `.eval()` and are affected.
- `vjepa2.py`: the action-conditioned predictor does not use the action (zero-action loss equals
  the true-action loss).
- `cjepa.py`: the collision step leaves two digits on one cell in about 5% of clips.
- The committed `samples/` figures predate the last rewrite of the algorithm files.
