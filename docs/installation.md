# Installation Guide

## Requirements

- Python 3.10 or 3.11
- ~5GB free disk for dependencies (PyTorch CPU wheel + transformers stack)
- macOS/Linux (Windows via WSL)

## Setup

```bash
git clone <repo-url>
cd brandparadigm
python3 -m venv .venv
source .venv/bin/activate

make install_dev   # installs runtime + dev dependencies, then `pip install -e .`
```

`make install` uses the CPU wheel index for PyTorch
(`https://download.pytorch.org/whl/cpu`) so it doesn't pull multi-gigabyte
CUDA libraries on machines without a GPU. Override with
`make install TORCH_INDEX=https://download.pytorch.org/whl/cu121` on a CUDA
machine.

## Environment variables

```bash
cp .env.sample .env
```

Edit `.env` to point at your own model/data locations if they differ from
the defaults (see `brandparadigm/config/settings.py`).

## Verify the install

```bash
python -c "import brandparadigm; print(brandparadigm.__version__)"
make lint
make test
```

## Next steps

- `docs/dataset_guide.md` — pulling and validating the three datasets.
- `docs/developer_guide.md` — package structure and coding conventions.
- `docs/deployment_guide.md` — running the API/dashboard/Docker image.
