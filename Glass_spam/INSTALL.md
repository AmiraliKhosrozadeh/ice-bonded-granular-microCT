# SPAM install — WSL Ubuntu 22.04 + Python 3.11

SPAM does not support native Windows (per upstream docs). This recipe installs
it in WSL Ubuntu 22.04 with a dedicated Python 3.11 venv. Do **not** install
it into the Dragonfly or system Python.

## 1. Install WSL Ubuntu (one-time, Windows side)

Open **Windows PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted. After reboot, Ubuntu asks for a UNIX username + password.

If `wsl` is already installed but Ubuntu 22.04 isn't:

```powershell
wsl --list --online
wsl --install -d Ubuntu-22.04
```

## 2. System packages (inside Ubuntu shell)

```bash
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 \
    git build-essential
```

## 3. Create a dedicated venv

```bash
python3.11 -m venv ~/spam-env
source ~/spam-env/bin/activate
pip install --upgrade pip wheel
```

## 4. Install SPAM and common dependencies

```bash
pip install spam numpy scipy scikit-image matplotlib tifffile pandas
```

## 5. Sanity check

```bash
python - <<'PY'
import spam, spam.DIC, spam.label
print('SPAM version:', spam.__version__)
print('DIC module :', spam.DIC.__name__)
print('label module:', spam.label.__name__)
PY
```

If this prints the version and two module names without traceback, SPAM is
ready.

## 6. Accessing the data

The staged TIFFs live under Windows:
`E:\RPTU-images\CT_images\Glass\Glass_spam\data\`

In WSL this path is mounted at:
`/mnt/e/RPTU-images/CT_images/Glass/Glass_spam/data/`

For speed, copy the files to a native Linux path before running SPAM — I/O
over `/mnt/e/...` is much slower than over `~/`:

```bash
mkdir -p ~/spam-data
cp /mnt/e/RPTU-images/CT_images/Glass/Glass_spam/data/*.tif ~/spam-data/
```

Then in `spam_analysis.py` point `DATA_DIR` to `~/spam-data`.

## Fallback: Docker

If `pip install spam` fails for your Python/OS combination, use the official
SPAM Docker image. Pull and run from WSL or from Docker Desktop on Windows:

```bash
docker pull registry.gitlab.com/spam-project/spam:latest
docker run --rm -it \
  -v /mnt/e/RPTU-images/CT_images/Glass/Glass_spam:/work \
  registry.gitlab.com/spam-project/spam:latest \
  bash
# Inside the container, SPAM is pre-installed. cd /work and run commands.
```

(Check the SPAM project's gitlab / docs for the current Docker image URL —
this one is a typical pattern but may need updating.)

## Activating the env for every session

```bash
source ~/spam-env/bin/activate
```

Confirm with `which python` → should show `~/spam-env/bin/python`.
