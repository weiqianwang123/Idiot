# Experiment Config

Each JSON file stores one model family.

The common shape is:

```json
{
  "model": {},
  "train": {},
  "eval": {}
}
```

Run a mode by passing the same config file to that model script:

```bash
python experiment/vae.py train --config experiment/config/vae.json
python experiment/vae.py eval --config experiment/config/vae.json
```
