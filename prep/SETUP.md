# Open4DGS-Core preparation setup

This guide covers the Python and system dependencies used by Open4DGS-Core
scripts under `prep/`. Open4DGS-Trigs has a separate trainer setup guide in
[Open4DGS-Trigs setup](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md).
For checkout layout and shared helper dependencies, start with [the combined setup](../SETUP.md).

## Choose an environment

The main preparation scripts use Python, NumPy, OpenCV, PyTorch, and model
packages. RoMa v1, RoMa v2, and MEMFOF are optional backends; install only the
ones needed for a run. Some larger workflows use incompatible framework
versions and should keep their own environments (listed below).

The shared environment shown here matches the current TriGS machine setup:
Ubuntu 24.04, Python 3.11, NVIDIA driver 595.84, Conda CUDA toolkit 12.8, and
PyTorch 2.10.0+cu128. It was smoke-tested on an RTX 3090. Each host needs its
own environment and any native extensions rebuilt for its GPU. Notes from
one [Windows 11 machine](#windows-11-with-nvidia-gpus) follow the Linux setup.

## 1. Install system tools

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git ninja-build curl ca-certificates \
  libgl1 libglib2.0-0 ffmpeg tmux
```

The scripts that process video call `ffmpeg` and `ffprobe`; both are provided
by Ubuntu's `ffmpeg` package. Prep alignment and reconstruction workflows also
use the standalone `colmap` executable. COLMAP is installed in step 3 below;
`pycolmap` is a separate Python package and does not provide this executable.
Check the executable after installation with:

```bash
command -v ffmpeg
command -v ffprobe
command -v colmap
```

## 2. Create the shared Python environment

Use Miniforge/Conda and Python 3.11. If Conda is not installed, install
[Miniforge](https://github.com/conda-forge/miniforge) on x86_64 Ubuntu:

```bash
curl -L -o "$HOME/Miniforge3.sh" \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash "$HOME/Miniforge3.sh" -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

If you already created the `trigs` environment while setting up the training
project, you can use it here too. Otherwise:

```bash
conda create -n trigs python=3.11 pip -y
conda activate trigs
```

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Install the CUDA 12.8 compiler toolkit for packages that build CUDA
extensions. The PyTorch wheel also brings its CUDA runtime; these serve
different purposes.

```bash
conda install -y nvidia/label/cuda-12.8.0::cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
```

Use a machine-appropriate extension target when installing packages that
compile CUDA code:

```bash
export TORCH_CUDA_ARCH_LIST=8.6   # RTX 3090 / Ampere
# For GeForce RTX 50-series Blackwell, use: export TORCH_CUDA_ARCH_LIST=12.0
```

Install PyTorch/torchvision from the CUDA 12.8 wheel index, then the common
preparation packages:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "$CONDA_PREFIX/bin/python" \
  numpy scipy pillow opencv-python tqdm imageio matplotlib pycolmap
```

Install additional packages only when a particular script needs them. The
TriGS environment has the broader set of project dependencies declared in its
own `pyproject.toml` and lockfile.

## 3. Install the COLMAP command-line program

**Validation status:** the COLMAP build on the Ubuntu 24.04 RTX 3090 host has
not completed. Configuration with Conda's GCC 14 succeeded, but compilation
failed because that compiler did not search `/usr/include/glog`. The procedure
below switches to Ubuntu's GCC in a fresh build directory; it is a proposed
correction and must be verified on that host before this section is considered
a working setup recipe.

Prep alignment and reconstruction workflows need the standalone COLMAP
executable. Use **COLMAP 3.12.0 with CUDA**, pinned to release commit
`e7e89eb0c82eccc00b78086e419f4def4e5860ae` (short id `e7e89eb0`). The Ubuntu
repository package may be a different build and may lack CUDA, so build the
requested release from source. These steps install the binary into the active
`trigs` environment; activate that environment when running prep commands.

The package list follows COLMAP's [3.12 Ubuntu build guide](https://colmap.github.io/legacy/3.12/install.html).
This is a headless CLI build, so Qt packages are not needed. Use the CUDA 12.8
toolkit already installed in Conda rather than installing a second system
toolkit:

```bash
mkdir -p "$HOME/src"
git clone --branch 3.12.0 --depth 1 \
  https://github.com/colmap/colmap.git "$HOME/src/colmap-3.12.0"
git -C "$HOME/src/colmap-3.12.0" rev-parse HEAD
```

Confirm that the printed commit is exactly
`e7e89eb0c82eccc00b78086e419f4def4e5860ae` before building. Then install the
system development libraries and build into the active Conda environment.
Before installing, refresh APT and make sure the Ubuntu 24.04 repositories are
consistent. The default Ubuntu deb822 configuration in
`/etc/apt/sources.list.d/ubuntu.sources` includes `noble` and `noble-updates`
in its main stanza, and `noble-security` in its security stanza. Keep the
machine's existing mirror and component settings; do not mix package versions
from different Ubuntu releases.

```bash
sudo apt-get update
apt-cache policy libgmp-dev libgmp10 libicu-dev libicu74 libmpfr-dev libmpfr6
```

The development package candidates and their corresponding runtime libraries
must come from compatible repository revisions. If APT reports that, for
example, `libgmp-dev` requires `libgmp10` at one exact version while a newer
`libgmp10` is selected, inspect the configured sources before retrying:

```bash
grep -R -n -E '^(Types:|URIs:|Suites:|Components:|deb )' \
  /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null
```

If `noble-updates` is missing from the Ubuntu source configuration, add it to
the existing Noble main source stanza (for example, change
`Suites: noble` to `Suites: noble noble-updates noble-backports`), run
`sudo apt-get update`, and check the package policies again. Do not force a
downgrade or install packages from a different Ubuntu release. If the
candidates still disagree, share the policy and source output with the system
administrator before continuing.

Once the candidates are compatible, install the build dependencies:

```bash
sudo apt-get install -y \
  cmake ninja-build build-essential git \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libopengl-dev libglx-dev \
  libeigen3-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev \
  libsqlite3-dev libglew-dev libceres-dev libsuitesparse-dev \
  libcurl4-openssl-dev libssl-dev libopenblas-openmp-dev
```

The OpenGL development packages are required at configure time even with the
COLMAP GUI disabled. The explicit Boost and OpenGL paths in the script make CMake use
the Ubuntu headers and component libraries together, rather than relying on
discovery through the Conda toolchain. The Ceres package directory is also
given explicitly because Ubuntu installs its CMake config under the system
architecture-specific CMake directory. Ubuntu's OpenMP OpenBLAS package keeps
its development symlink in a vendor subdirectory and its shared library
provides both BLAS and LAPACK, so pass that path for both checks. Ceres' bundled
SuiteSparse finder searches each component separately; the explicit library
paths in the script cover its required components.
The explicit METIS release path is needed because Ceres creates a METIS target
from the system header even when the Conda toolchain cannot discover its
library; that target needs a concrete location for CMake's link check.
Ubuntu's Ceres package also exports gflags and glog as CMake dependencies;
their package directories are specified so CMake loads the matching Ubuntu
targets. The script uses Ubuntu's GCC so it searches the matching Ubuntu
headers, including `/usr/include/glog`.
This prep build disables COLMAP's optional CGAL integration. The repository's
prep scripts use feature extraction, matching, sparse mapping, and model
conversion; none invokes COLMAP's CGAL-based Delaunay mesher. Avoiding CGAL also
avoids its additional GMP/MPFR configuration in this mixed Conda/system build.
The script requests CUDA. PoseLib and FAISS use COLMAP's default bundled
FetchContent mode, so the build machine needs GitHub access.
The build script stops if a step fails; running it with `bash` does not change
the error behavior of the interactive tmux shell.

Run the repository script from the repository root after activating the environment.
It checks the pinned source commit, required headers and libraries, and the
compiler choice before configuring. It uses a fresh system-compiler build
directory, stops on the first error without exiting the interactive tmux shell,
and verifies that the installed CLI reports version 3.12.0 with CUDA.

```bash
conda activate trigs
bash prep/install-colmap-3.12-cuda.sh
```

For an RTX 5090/other GeForce RTX 50-series card, set
`COLMAP_CUDA_ARCHITECTURES=120`. For other GPU models, use the numeric target
matching their compute capability. Rebuild COLMAP on each machine for its GPU.

Verify the selected binary and build features:

```bash
hash -r
command -v colmap
colmap help
```

`command -v colmap` should resolve to `$CONDA_PREFIX/bin/colmap`; the help
header should report `COLMAP 3.12.0` and `with CUDA`. The release tag and
commit are available at the [COLMAP 3.12.0 release commit](https://github.com/colmap/colmap/commit/e7e89eb0c82eccc00b78086e419f4def4e5860ae).

### Dependency inventory by workflow

This inventory is based on Python imports and setup instructions in the
repository. It identifies packages users need to account for; it does not
claim that every optional model has been downloaded or exercised on every GPU.

| Workflow | Python packages | System tools / notes |
| --- | --- | --- |
| General prep utilities (`prep/init`, `prep/align`, `prep/sync`, masks) | `numpy`, `scipy`, `pillow`, `opencv-python`, `tqdm`; GPU/model scripts also need `torch` and `torchvision` | `ffmpeg` and `ffprobe` for video; COLMAP 3.12.0 CLI with CUDA is installed in step 3; `pycolmap` is a separate Python package |
| RoMa spatial matching | Choose `romatch` (v1) or `romav2` (v2) | Both use PyTorch; model weights download at first use. RoMa v1 has an optional acceleration package; RoMa v2 declares its Linux local-correlation dependency |
| Memory initialization (`prep/init/prep-roma-trigs-memory.py`) | Core: `numpy`, `scipy`, `pillow`, `opencv-python`, `torch`, `tqdm`; spatial: `romatch` or `romav2`; optional temporal: `memfof` | No external executable. MEMFOF installs `einops`, `huggingface-hub`, `safetensors`, and requires `--centred-velocity`; its Tartan-T-TSKH checkpoint downloads from Hugging Face. |
| BGMv2 production masks | Shared stack: `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`, `tqdm`; `gdown` only for the documented checkpoint download | Requires a clean background plate and the BGMv2 TorchScript checkpoint. The raw-weight fallback also needs a BGMv2 source checkout. |
| SAM2 automatic masks | Shared stack plus editable `SAM-2`, which installs `hydra-core` and `iopath` | Builds an optional CUDA extension and needs the `sam2.1_hiera_large.pt` checkpoint. |
| VGGT alignment workflow | See its documented environment and install path | [VGGT guide](https://github.com/facebookresearch/vggt); uses downloaded model weights |
| EasyVolCap / 4K4D camera refinement | Separate Python 3.10 environment with the exact Torch 2.1.2 / CUDA 11.8 / PyTorch3D 0.7.8 stack | [EasyVolCap guide](easyvolcap/README.md); also builds tiny-cuda-nn |
| Repository tests | `pytest` plus the relevant workflow packages | Tests live in corresponding `tests/` directories; install workflow-specific dependencies before running them |

The TriGS training project has its own `pyproject.toml` and lockfile for exact
resolved versions. Fused SSIM is an optional native extension installed
separately by that project's setup guide.

## 4. Install RoMa and MEMFOF as needed

`prep/init/prep-roma-trigs-memory.py` has no executable dependencies beyond
the packages in this section. Its direct Python imports are `numpy`, `scipy`,
`pillow`, `opencv-python`, `torch`, and `tqdm`; step 2 installs all six. The
script also imports the repository-local `rigid_motion.py`, `motion_budget.py`,
and `roma_backends.py`, which introduce no additional third-party packages.

It needs one RoMa backend for spatial correspondences and, with the default
settings, MEMFOF for temporal tracks. The defaults are RoMa v2
(`--roma-backend v2`) and MEMFOF (`--temporal-backend memfof`); install both to
run the default pipeline. RoMa v1 and RoMa-only temporal tracking remain
available through flags.

### RoMa v1

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" romatch
```

Select it with `--roma-backend v1` (the default is v2). The package resolver also
installs RoMa v1's declared dependencies: `albumentations`, `einops`, `h5py`,
`kornia`, `loguru`, `matplotlib`, `poselib`, `timm`, and `wandb`; shared
packages such as Torch, torchvision, OpenCV, and tqdm remain in the existing
environment. RoMa v1 can optionally use `fused-local-corr` for acceleration;
its adapter has a PyTorch fallback. Upstream instructions:
[RoMa](https://github.com/Parskatt/RoMa).

On Windows, RoMa v1 uses a slower PyTorch local-correlation fallback. One
reported `romatch` 0.1.2 installation also spent substantial time formatting
a CUDA tensor during matching. If matching is unexpectedly slow, check the
installed version and upstream issue status; performance depends on the backend
and platform. RoMa v1 is selected with `--roma-backend v1`.

### RoMa v2

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" \
  "git+https://github.com/Parskatt/RoMaV2.git@v2.0.1"
```

RoMa v2 is the script's default backend. It installs `einops`, `rich`, and its
Linux `fused-local-corr` dependency; the latter is a native CUDA extension.
It also requires the already-installed Torch, torchvision, Pillow, and tqdm.
First use downloads model weights. Review the upstream repository's code and
model licenses before use, including its DINOv3 license terms:
[RoMaV2](https://github.com/Parskatt/RoMaV2).

### MEMFOF temporal flow

MEMFOF is the default temporal-flow backend; RoMa remains in use for spatial
matching. This repository defaults to the upstream real-video checkpoint
`egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH`. “Tartan” is part of the
checkpoint's training lineage; running inference does **not** require
downloading the TartanAir dataset.

Install the upstream package at the currently documented `dev`-branch commit:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" \
  "git+https://github.com/msu-video-group/memfof.git@a51de9fc59c6fe20ba08e079372c7b583d58a712"
```

MEMFOF declares and installs `einops`, `huggingface-hub`, and `safetensors`,
in addition to its supported `torch` and `torchvision` versions. The shared
environment already supplies Torch 2.10/cu128 and torchvision 0.25/cu128,
which satisfy MEMFOF's `torch>=2.5` and `torchvision>=0.20` requirements. The
Tartan-T-TSKH weights download from Hugging Face when the model is first
loaded. If the machine has an ephemeral home directory, set `HF_HOME` to a
directory on its mounted persistent volume; the default is
`$HOME/.cache/huggingface`.

```bash
export HF_HOME="/path/on/persistent-volume/huggingface"
```

MEMFOF needs temporal tracking and centred velocities, which are both on by
default. To track with RoMa instead, pass
`--temporal-backend roma`. See the script's `--help`
for the required dataset and output arguments, and
[docs/run-pipeline.md](../docs/run-pipeline.md) for the full pipeline commands. Upstream install and checkpoint
information: [MEMFOF](https://github.com/msu-video-group/memfof).

### Verify the exact memory-init stack

The following command checks every third-party import used by
`prep-roma-trigs-memory.py`, loads the exact Tartan-T-TSKH checkpoint, and
runs a small CUDA inference. It downloads the checkpoint on its first run.
Run it after installing RoMa and MEMFOF:

```bash
python - <<'PY'
import cv2
import einops
import numpy as np
import safetensors
import scipy
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from memfof import MEMFOF

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
checkpoint = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
model = MEMFOF.from_pretrained(checkpoint).eval().to("cuda")
inputs = torch.zeros((1, 3, 3, 256, 256), device="cuda")
with torch.inference_mode():
    backward, forward = model(inputs)["flow"][-1].unbind(dim=1)
assert backward.shape == forward.shape == (1, 2, 256, 256)
print("MEMFOF Tartan-T-TSKH CUDA inference: OK")
PY
```

Then load the selected RoMa model through the same adapter used by the prep
script. The first run downloads its RoMa weights:

```bash
# RoMa v1
python - <<'PY'
import sys
sys.path.insert(0, "prep/init")
from roma_backends import create_roma_backend
print(create_roma_backend(backend="v1", device="cuda").name)
PY

# RoMa v2 (the prep-script default)
python - <<'PY'
import sys
sys.path.insert(0, "prep/init")
from roma_backends import create_roma_backend
print(create_roma_backend(backend="v2", device="cuda").name)
PY
```

Run only the command for the backend selected with `--roma-backend`. The
adapter imports optional packages only when that backend is selected. These
commands confirm package imports and model loading; use a small input dataset
before a full preprocessing run to confirm your chosen data options.

## 5. Install mask-generation dependencies as needed

`prep/mask/generate_masks.py` always imports `numpy`, `torch`,
`opencv-python`, `tqdm`, and `pillow`; step 2 installs them. It has two
optional model paths. It launches a fresh Python worker per camera for BGMv2,
but does not call a separate external executable.

### BGMv2 production masks

This is the default when `data_dir/background/` contains clean plates. The
supported path uses the BGMv2 **TorchScript** ResNet-50 checkpoint and needs
no BGMv2 Python package or source checkout. Download it into the location
that the script searches by default:

```bash
conda activate trigs
uv pip install --python "$CONDA_PREFIX/bin/python" gdown
mkdir -p "$HOME/BackgroundMattingV2/checkpoints"
gdown https://drive.google.com/uc?id=1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf \
  -O "$HOME/BackgroundMattingV2/checkpoints/torchscript_resnet50_fp32.pth"
```

The source project publishes legacy package pins including Torch 1.7. Do not
install its `requirements.txt` into this shared PyTorch 2.10 environment. This
repo loads the TorchScript checkpoint directly with the already installed
Torch and torchvision. The source checkout is needed only if deliberately
using a non-TorchScript raw state dict. Keep it separate from the TorchScript
checkpoint directory and set `BGMV2_REPO` before invoking the mask script:

```bash
mkdir -p "$HOME/src"
git clone https://github.com/PeterL1n/BackgroundMattingV2.git \
  "$HOME/src/BackgroundMattingV2"
export BGMV2_REPO="$HOME/src/BackgroundMattingV2"
```

That raw-state-dict fallback has not been validated with the shared Torch 2.10
stack; use the TorchScript checkpoint above for the supported setup.

Verify the exact production model path and one CUDA inference:

```bash
python - <<'PY'
import os
import sys
import numpy as np
import torch

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
sys.path.insert(0, "prep/mask")
from generate_masks import load_bgmv2_model, run_bgmv2

checkpoint = os.path.expanduser(
    "~/BackgroundMattingV2/checkpoints/torchscript_resnet50_fp32.pth"
)
model = load_bgmv2_model(checkpoint, "cuda")
image = np.zeros((256, 256, 3), dtype=np.uint8)
alpha, foreground = run_bgmv2(model, image, image, "cuda")
assert alpha.shape == (256, 256)
assert foreground.shape == (256, 256, 3)
print("BGMv2 TorchScript CUDA inference: OK")
PY
```

The model is from the official
[Background Matting V2 release](https://github.com/PeterL1n/BackgroundMattingV2/releases/tag/v1.0.0).
It requires a background image for each camera; that data requirement is
separate from the model dependency.

### SAM2 automatic-mask fallback

Use this path with `--method sam2` when no clean background plates exist.
The current SAM2 package requires Python 3.10 or newer, Torch 2.5.1 or newer,
and torchvision 0.20.1 or newer. The shared environment meets those version
requirements. Its editable install adds `hydra-core` and `iopath` (and checks
the existing NumPy, Pillow, and tqdm versions); it builds the optional SAM2
CUDA connected-components extension against the active Torch/CUDA toolchain.

```bash
conda activate trigs
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3090 / Ampere

git clone https://github.com/facebookresearch/sam2.git "$HOME/sam2_repo"
uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation \
  -e "$HOME/sam2_repo"
curl -fL \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o "$HOME/sam2_repo/checkpoints/sam2.1_hiera_large.pt"
```

For an RTX 50-series Blackwell GPU, replace `8.6` with `12.0`. The SAM2
checkout must remain available because its installed package supplies the
`configs/sam2.1/sam2.1_hiera_l.yaml` configuration used by this script.

Verify imports, the exact checkpoint, configuration lookup, and one CUDA
model load before processing a full capture:

```bash
python - <<'PY'
import os
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
checkpoint = os.path.expanduser("~/sam2_repo/checkpoints/sam2.1_hiera_large.pt")
model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint, device="cuda")
SAM2AutomaticMaskGenerator(model=model)
print("SAM2.1 large CUDA model load: OK")
PY
```

If SAM2 cannot find its `configs/...` file after a successful editable
install, set the official fallback before rerunning it:

```bash
export PYTHONPATH="$HOME/sam2_repo:${PYTHONPATH}"
```

SAM2 package and checkpoint requirements are documented in the official
[SAM2 repository](https://github.com/facebookresearch/sam2) and
[installation guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).

## Other preparation workflows

These workflows have their own requirements and should not be installed into
the shared `trigs` environment unless their guide explicitly says so:

- **VGGT:** [docs/vggt-setup-and-run.md](https://github.com/facebookresearch/vggt)
  includes its environment, run instructions, and model-weight terms.
- **EasyVolCap / 4K4D:** [prep/easyvolcap/README.md](easyvolcap/README.md)
  documents its separate Python and CUDA stack.
- **BGMv2 and SAM2 masks:** see the complete install and validation section
  above.

Before adding a new package to a shared environment, check the workflow's
existing setup notes: the repo contains pipelines pinned to different Python,
PyTorch, and CUDA versions.

## Windows 11 with NVIDIA GPUs

These notes come from one Windows 11 machine with an RTX PRO 6000 Blackwell
Workstation Edition. They are not a validation of the preparation workflow on
Windows. Both repositories were at v0.1.0 and checked out side by side. The
environment was the trainer's `.venv`, built with the
[trainer's Windows recipe](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md#windows-11-with-nvidia-gpus)
with `set DISTUTILS_USE_SDK=1` added (Python 3.11, PyTorch 2.10.0+cu128). The
following programs ran natively, without WSL 2, to the end without errors:

- `prep/sync/calculate-sync.py`
- `prep/init/prep-roma-trigs-memory.py`, with the default backends (RoMa v2
  `fast` and MEMFOF) and `--timing-json`
- `deliver/make-trigs-timeline.py from-checkpoint`

Frame extraction, the mask tools, COLMAP and the bash scripts were not used.
With the checkouts side by side, initialization found the trainer's helper
files without `OPEN4DGS_TRIGS` set.

A trainer `uv sync` can remove separately installed packages, as the trainer's
[shared-environment notes](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md#sharing-an-environment-with-preparation-tools)
explain, so install the preparation backends after it. In Command Prompt, from
the trainer project directory:

```bat
set UV_CACHE_DIR=C:\uv-cache
uv pip install --python .venv\Scripts\python.exe ^
  "git+https://github.com/Parskatt/RoMaV2.git@v2.0.1"
uv pip install --python .venv\Scripts\python.exe ^
  "git+https://github.com/msu-video-group/memfof.git@a51de9fc59c6fe20ba08e079372c7b583d58a712"
```

RoMa v2 marks its `fused-local-corr` requirement as Linux-only, so it is not
installed on Windows. RoMa v2 then uses its PyTorch local-correlation path, and
no CUDA extension is compiled for these two packages.

The commands in this guide, the README and `docs/run-pipeline.md` are written
for bash. In Command Prompt, continue
lines with `^`, use `set` in place of `export`, write `$REPO` as `%REPO%`, and
call the environment's `python.exe` directly; a uv-created Windows environment
has no `python3`. For example, the initialization command:

```bat
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set REPO=C:\path\to\open4dgs-core
set TRIGS=C:\path\to\open4dgs-trigs
cd /d D:\path\to\capture
"%TRIGS%\.venv\Scripts\python.exe" "%REPO%\prep\init\prep-roma-trigs-memory.py" ^
  --reference-json reference_export.json --images-root images ^
  --frame-start <FIRST_FRAME> --frame-end <LAST_FRAME> ^
  --timing-json timing.json --output-path keyframes.npz
```

The two UTF-8 settings match the trainer recipe. Runs without them were not
tried.

## Blackwell GPUs

For a Blackwell card, first confirm the exact GPU and compute capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv` or
`torch.cuda.get_device_capability(0)`. GeForce RTX 50-series cards, including
the RTX 5090, use compute capability 12.0; other Blackwell product families
can use different targets. See NVIDIA's
[compute-capability table](https://developer.nvidia.com/cuda/gpus). For an RTX
50-series card, build CUDA extensions with `TORCH_CUDA_ARCH_LIST=12.0`; for
fused SSIM, use `CUDA_ARCHITECTURES=120` and install it after the PyTorch
environment is ready with `--no-build-isolation`. For the standalone COLMAP
build in step 3, set `COLMAP_CUDA_ARCHITECTURES=120`. Do not reuse extensions
built for an Ampere card. Confirm that the PyTorch CUDA wheel supports the
installed GPU and run a small inference before a full preprocessing job.
RoMa, MEMFOF, and the isolated preparation workflows have not all been
validated on Blackwell as a group.

## Remote work with tmux

Install tmux on each Ubuntu host with `sudo apt-get install tmux`. Nested tmux
sessions are valid when SSH hops require them, but each layer has its own
prefix and session. To enable mouse support on a host, add this setting to
`~/.tmux.conf` and reload it:

```bash
grep -qxF 'set -g mouse on' ~/.tmux.conf 2>/dev/null || \
  echo 'set -g mouse on' >> ~/.tmux.conf
tmux source-file ~/.tmux.conf
```

Mouse scrolling is handled by the innermost tmux session; use copy mode
(`Ctrl-b`, then `[`) to scroll in a particular session. Detach with `Ctrl-b`,
then `d`; this leaves that tmux session running as long as its server/host
remains alive. Avoid `set -euo pipefail` in an interactive shell because a
failed command can close the shell or pane.
