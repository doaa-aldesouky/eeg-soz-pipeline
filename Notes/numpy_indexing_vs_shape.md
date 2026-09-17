# NumPy Reference: Indexing/Slicing vs. .shape

A quick note to stop confusing these two — they answer completely
different questions.

## The core distinction

| | What it does | What it returns |
|---|---|---|
| `data[i]` or `data[:i]` | **Slicing** — reaches INTO the array and pulls out actual values | The real numbers/signal — a wall of data |
| `data.shape[i]` | **Inspecting** — asks about the array's SIZE, doesn't touch the values | A single integer describing structure |

If you want a *number describing the array*, you want `.shape`.
If you want *the actual values inside* the array, you want indexing/slicing.

## Concrete example

For EEG data with shape `(n_channels, n_samples)`:

```python
data.shape        # (23, 921600)  <- a tuple: (num_channels, num_samples)
data.shape[0]     # 23            <- the number of channels
data.shape[1]     # 921600        <- the number of samples

data[0]           # channel 0's full signal, as a 1D array of 921600 numbers
data[1]           # channel 1's full signal, as a 1D array of 921600 numbers
data[:1]          # channel 0's signal, but kept 2D: shape (1, 921600)
```

## The question to ask yourself

> "Do I want to know something ABOUT the array (its size, its shape),
> or do I want the actual DATA stored inside it?"

- About the array → `.shape`
- The data itself → indexing (`[i]`) or slicing (`[:i]`, `[i:j]`)

## One more distinction worth remembering: `[i]` vs `[:i]`

Even within slicing, these aren't the same:

```python
data[0]     # shape (921600,)   <- 1D, that dimension is DROPPED
data[:1]    # shape (1, 921600) <- 2D, that dimension is KEPT (just size 1)
```

`[i]` collapses a dimension. `[:i]` (or `[i:j]`) keeps the array's
original number of dimensions, just with fewer entries along that axis.

This matters in practice because some functions expect a specific
number of dimensions — passing `data[0]` where a 2D array was expected
can silently break things or throw a confusing shape-mismatch error.
