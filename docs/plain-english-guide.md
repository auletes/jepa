# The JEPA models in plain English

A starting point for readers who have not worked with these models before. It explains what each
of the six programs in this repository does, without equations. The tutorials (`*_tutorial.md`)
go deeper once this makes sense.

## The one idea behind all six

Hide part of an image or video from a model, and ask it to guess what is in the hidden part.

The twist is in *what* it guesses. It does not try to paint the missing pixels. It guesses a
**summary** of the missing part: a short list of numbers that describes what is there. This
summary is called an **embedding**.

Why guess a summary and not pixels? Because most pixel detail is unpredictable and unimportant.
If a dog's head is hidden, nobody can guess the exact position of each hair, but "a dog's head,
facing left" is guessable. Guessing summaries pushes the model to learn the meaningful part and
ignore the noise.

No labels are involved. Nobody tells the model "this is a dog". It learns only by playing this
hide-and-guess game millions of times. That is what "self-supervised" means.

The name spells out the same idea: **J**oint **E**mbedding **P**redictive **A**rchitecture is a
model that *predicts embeddings*.

## The three parts

Every model here is built from the same three pieces.

| Part | Job | Analogy |
|---|---|---|
| **Context encoder** | Looks at the visible part and writes a summary of each piece | A student reading a page with some paragraphs blacked out |
| **Target encoder** | Looks at the whole thing and writes the "correct" summaries | The answer key |
| **Predictor** | Takes the student's summaries and guesses the summaries of the hidden pieces | The student's answers |

Training compares the student's answers with the answer key and nudges the student to do better
next time. The number printed as `loss` during training is how far apart they are. Lower is
better.

Before any of this, the input is cut into small squares called **patches** (for video, small
boxes spanning two frames, called **tubelets**). The model works on these pieces, not on
individual pixels. Hiding part of the input is called **masking**.

## The trap every model must avoid

There is an easy way to cheat. If both encoders output the same summary for every input, say all
zeros, the student's guess is always "correct" and the loss is perfect. The model has learned
nothing. This failure is called **collapse**.

Each model has a way to prevent it, and this is the main thing that differs between them:

- **A slow-moving answer key.** The target encoder is not trained directly. It is a slowly
  updated average of the context encoder (called an **EMA**, exponential moving average). Because
  it lags behind, the two cannot quietly agree on a useless answer. I-JEPA, V-JEPA, and
  V-JEPA 2.1 do this.
- **A frozen answer key.** The summaries come from something that never changes during training.
  V-JEPA 2's second phase and C-JEPA do this.
- **A rule about the shape of the summaries.** An extra penalty forces the summaries to stay
  spread out, so they cannot all be the same. LeWorldModel does this.

## The six models

### 1. I-JEPA: images (`ijepa.py`)

- **Sees:** a small photo (CIFAR-10, 32×32 pixels) cut into an 8×8 grid of 64 patches.
- **Hidden:** four rectangular blocks of patches.
- **Guesses:** the summary of each hidden block, given one large visible block and the positions
  of the hidden ones.
- **Why it matters:** this is the original recipe. Everything else is a variation on it.

After training, the encoder's summaries are useful on their own. A very simple classifier placed
on top of them can tell the ten CIFAR-10 categories apart, even though the encoder never saw a
label. That check is called a **linear probe**.

### 2. V-JEPA: video (`vjepa.py`)

- **Sees:** a 10-frame clip of two handwritten digits drifting around (Moving MNIST).
- **Hidden:** the same region of the screen across *every* frame, like a pillar standing in
  front of the camera.
- **Guesses:** the summaries of everything behind the pillar.
- **What is new:** if only one frame were hidden, the model could copy from the frame before.
  Hiding the region for the whole clip forces it to understand motion: where the digit went
  while it was out of sight.

### 3. V-JEPA 2: video, then actions (`vjepa2.py`)

Two phases.

- **Phase 1** is V-JEPA again. It produces an encoder that summarises video well.
- **Phase 2** freezes that encoder and trains a second, separate predictor. This one is told
  what **action** was taken (here, the velocity applied to a digit) and must guess the summary of
  the *next* moment.

Phase 2 turns the model into a **world model**: something that answers "if I do this, what
happens next?". Chaining its guesses to look several steps ahead is called a **rollout**. A robot
could use this to try out plans in its head before moving.

Caveat: in this toy version the predictor learns to ignore the action. See `STUDY.md`, finding M1.

### 4. V-JEPA 2.1: sharper features (`vjepa2_1.py`)

Same game as V-JEPA, with three refinements:

- The model must also reproduce the summaries of the pieces it *can* see, not only the hidden
  ones. This keeps every patch's summary precise about its own location.
- Summaries are checked at four depths inside the encoder, not only at the end.
- One encoder handles both still images and videos, trained on alternating batches of each.

The goal is summaries that are accurate patch by patch, which matters for tasks like outlining
objects or tracking them.

### 5. C-JEPA: objects (`cjepa.py`)

- **Sees:** three digits bouncing around a 4×4 board for 8 frames. The scene is described as
  three **slots**, one per object, each holding that object's position.
- **Hidden:** one or two whole objects for frames 1 to 4, and all objects for frames 5 to 7. In
  frame 0 everything is visible, so the model knows which object is which.
- **Guesses:** where the hidden objects were, and where every object will go.
- **What is new:** hiding whole objects, not screen regions. To fill in a hidden object's path,
  the model has to reason about how the visible objects behaved, for example noticing that one of
  them bounced off something it cannot see.

Simplification: the real method first has to discover the objects in raw video. This version is
handed the true positions, so it teaches only the hide-and-guess part.

### 6. LeWorldModel: the simplest world model (`leworldmodel.py`)

- **Sees:** a clip of one moving digit, with the action (velocity) for each frame.
- **Guesses:** the summary of the next frame from the frames and actions so far.
- **What is new:** there is no separate answer key at all. One encoder produces both the inputs
  and the targets, and it is trained together with the predictor.

That setup would collapse immediately, so a second penalty is added, called **SIGReg**. It checks
that the summaries across a batch are spread out like a bell curve in every direction. Summaries
that are all the same would fail this check badly, so the model cannot cheat.

## Side by side

| Model | Input | What is hidden | How collapse is avoided |
|---|---|---|---|
| I-JEPA | Photo | Rectangular blocks | Slow-moving answer key |
| V-JEPA | Video | Regions hidden for the whole clip | Slow-moving answer key |
| V-JEPA 2 | Video + actions | Phase 1: as V-JEPA. Phase 2: the next moment | Slow-moving, then frozen |
| V-JEPA 2.1 | Photos and video | As V-JEPA, plus visible pieces are checked too | Slow-moving answer key |
| C-JEPA | Object positions | Whole objects, and the future | Frozen answer key |
| LeWorldModel | Video + actions | The next frame | Rule on the shape of the summaries |

## What you see when you run one

```
ep=4 step=  950 loss=0.0732 lr=1.09e-04 ema=0.9984
```

| Field | Meaning |
|---|---|
| `ep` | Epoch: how many times the model has gone through the whole dataset |
| `step` | Number of training updates so far |
| `loss` | Distance between the guesses and the answer key. It should fall over time |
| `lr` | Learning rate: how large each correction is. It rises briefly, then shrinks |
| `ema` | How slowly the answer key follows the student. Closer to 1 means slower |

Some models print extra numbers:

- `short` / `long` (V-JEPA family): the loss for small hidden regions and for large ones.
- `tf` / `roll` (V-JEPA 2 phase 2): the loss when guessing one step ahead from true history, and
  when chaining two guesses.
- `a=0` and `gap` (V-JEPA 2, LeWorldModel): the loss when the action is replaced by zero, and how
  much worse that is. A clear positive gap means the model really uses the action.
- `hist` / `fut` (C-JEPA): the loss for filling in hidden history and for forecasting.
- `sigreg` (LeWorldModel): how far the summaries are from the bell-curve shape.

A low loss alone does not prove the model learned something useful, since a collapsed model also
has a low loss. That is why the `*_extras.py` scripts add independent checks such as the linear
probe and the action gap.

## What these models do not do

- They do not generate images, video, or text. Their output is summaries.
- The versions here are tiny and train in minutes on a laptop. They show how the methods work.
  They are not the large pretrained models published with the papers.

## Where to go next

1. Run `cjepa.py` (about 10 seconds per epoch) to see training output with nothing to download
   beyond MNIST.
2. Read `ijepa_tutorial.md` alongside `ijepa.py`. The other five build on it.
3. Follow the order V-JEPA, V-JEPA 2, V-JEPA 2.1, C-JEPA, LeWorldModel. Each tutorial explains
   only what changed from the previous one.
4. Read `FAITHFULNESS.md` for what each file simplifies, and `STUDY.md` for known bugs.

## Glossary

| Term | Meaning |
|---|---|
| Embedding | A list of numbers that summarises a piece of the input |
| Encoder | The network that turns patches into embeddings |
| Predictor | The network that guesses the embeddings of hidden pieces |
| Patch / tubelet | A small square of an image / a small box of video spanning two frames |
| Masking | Hiding part of the input from the model |
| Context / target | The visible part / the hidden part to be guessed |
| Loss | How wrong the guesses are; training tries to reduce it |
| Collapse | The failure where every input gets the same embedding |
| EMA | A slowly updated average copy of a network, used as the answer key |
| Stop-gradient | Treating the answer key as fixed while correcting the student |
| Latent | Another word for "in embedding form", as in "latent space" |
| World model | A model that predicts what happens next, given an action |
| Rollout | Chaining predictions to look several steps ahead |
| Linear probe | A simple classifier on top of frozen embeddings, used to test their quality |
| Slot | An embedding that stands for one object |
| SIGReg | LeWorldModel's penalty that keeps embeddings spread out like a bell curve |
