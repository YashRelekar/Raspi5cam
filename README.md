# Raspi5cam

This repository is being rebuilt bit by bit.

The previous implementation has been preserved under [`old/`](old/) for reference,
including the original installer, Python source, models, and configuration.

## New structure (in progress)

```
Raspi5cam/
├── old/                  # prior implementation (reference only)
├── src/
│   └── raspi5cam/
│       ├── __init__.py
│       └── __main__.py
├── scripts/
│   └── install.sh        # placeholder – install steps being rebuilt
├── pyproject.toml
└── README.md             # this file
```

## Quick start

```bash
pip install -e .
python -m raspi5cam
```

## Prior content

All original files (scripts, requirements, models, source code) live under
[`old/`](old/).  Refer to [`old/README.md`](old/README.md) for the original
documentation.
