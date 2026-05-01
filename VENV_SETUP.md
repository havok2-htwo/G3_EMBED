# G3_EMBED Venv Setup

The recommended entrypoints are:

Windows:

```bat
start.bat
```

Linux/macOS:

```bash
bash ./start.sh
```

Both scripts create `venv` if needed, install PyTorch through `tools/install_torch_backend.py`, install `requirements.txt`, install frontend dependencies, build the frontend, and then launch `backend.genesis_embed_server`.

Manual setup:

```bash
python -m venv venv
venv/bin/python -m pip install --upgrade pip setuptools wheel
venv/bin/python tools/install_torch_backend.py --python venv/bin/python
venv/bin/pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
venv/bin/python -m backend.genesis_embed_server
```

On Windows, use `venv\Scripts\python.exe` and `venv\Scripts\pip.exe`.
