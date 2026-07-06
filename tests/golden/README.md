# Golden Vectors

Golden vectors freeze representative Albucore public-router behavior on small deterministic arrays.

Regenerate explicitly:

```bash
uv run --extra headless python tools/generate_golden_vectors.py
```

Verify:

```bash
uv run --extra headless python tools/verify_golden_vectors.py
```

Tests never regenerate goldens automatically.
