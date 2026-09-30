# `torch.export` freezes the batch size - and refuses a dynamic batch if the example has 1

**Symptom.** Exported the gesture CNN with `torch.zeros(1, 3, 128, 128)` as the example; the
reloaded model then failed on the validation set:

```
AssertionError: Guard failed: input.size()[0] == 1
```

Adding `dynamic_shapes={"input": {0: torch.export.Dim.DYNAMIC}}` with the same batch-1 example
failed at export time instead:

```
ValueError: Found the following conflicts between user-specified ranges and inferred ranges from model tracing:
- Received user-specified dim hint Dim.DYNAMIC(min ...
```

**Cause.** Export specializes every shape of the example unless told otherwise, and it treats a
size of 0 or 1 as a constant even when the dimension is marked dynamic.

**Fix, measured (torch 2.9.0+cpu).** An example batch of **2** plus a dynamic batch dimension:

```python
torch.export.export(model, (torch.zeros(2, 3, 128, 128),), dynamic_shapes=({0: torch.export.Dim.DYNAMIC},))
```

```
batch 1, Dim.DYNAMIC: FAILED ValueError ...
batch 2, Dim.DYNAMIC: works for batch sizes {1: True, 7: True, 500: True}   (allclose to the original)
```

**Consequences.**

- The exported model runs at batch 1 (the game) and at any other size (the notebook's checks).
- Use the tuple form `({0: Dim.DYNAMIC},)`: the dict form needs the forward argument's name,
  which is `input` for `nn.Sequential` and `x` for a hand-written module.
- `torch.export.load(path).module()` needs no class definition at load time - which is why a model
  defined in a notebook can be played by `play_snake.py`. `torch.save(model)` could not do that.
