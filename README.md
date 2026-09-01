# recurrent-optimization

Looping a single MLP to learn functions over increasingly large domains.

| Record | Max $b$ | Recurrent depth | Description |
|---:|---:|---:|---|
| [0](records/000_no_normalization.py) | $8\pi$ | 8 | No normalization |
| [1](experiment.py) | $24\pi$ | 8 | Fixed soft RMS clipping, $\tau=4$ |

```bash
uv run experiment.py
```
