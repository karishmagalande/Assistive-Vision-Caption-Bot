Trained model files go in `artifacts/model/` after you run `scripts/train.py`:

- `config.json`, `vocab.json`, `split.json`
- `decoder.weights.h5`
- `encoder.weights.h5` (only when the CNN was fine tuned)
- `history.json`, `metrics.json`

They are small enough to commit to GitHub (well under 100 MB) so the Docker image can be built straight from the repo.
