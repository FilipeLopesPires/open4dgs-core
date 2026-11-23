# Install Open4DGS-Core and Open4DGS-Trigs

The repositories are separate; the main pipeline can use one Python environment.
Core contains preparation and delivery scripts under MIT. The trainer is AGPL-3.0
and retains its third-party notices. Its import name and commands remain `trigs`.

## 1. Clone both projects side by side

```bash
git clone https://github.com/ASU-NEM-Labs/open4dgs-core.git
git clone https://github.com/ASU-NEM-Labs/open4dgs-trigs.git
```

Keep the checkout names above. Some preparation tools load the trainer's
lightweight capture-timing and COLMAP helpers. They find the sibling checkout
without installing CUDA training extensions. For a different location, set
`OPEN4DGS_TRIGS=/absolute/path/to/open4dgs-trigs`. An installed `trigs` package
also works. The working repository's `train/trigs/` layout remains supported.
These helpers stay in the trainer repository under their existing licenses.

## 2. Install the trainer first

Follow [Open4DGS-Trigs setup](https://github.com/ASU-NEM-Labs/open4dgs-trigs/blob/main/SETUP.md)
to create the `trigs` Conda environment, install the CUDA toolkit and build tools,
and set the architecture for your GPU. From the trainer checkout, with that
environment active:

```bash
cd open4dgs-trigs
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
uv sync --locked --inexact --preview-features extra-build-dependencies
```

The guide also includes Windows instructions. The documented main preparation
workflow uses Ubuntu; it has not been validated end to end on Windows.
[Notes from one Windows 11 machine](prep/SETUP.md#windows-11-with-nvidia-gpus)
list the programs that ran there.

## 3. Add preparation tools to the same environment

Follow [prep/SETUP.md](prep/SETUP.md) for COLMAP, common Python packages,
RoMa v2, MEMFOF, and your chosen mask backend. Reuse the activated `trigs`
environment; skip creating it again. Core is a collection of scripts, not a
second pip package. Model weights are downloaded separately.

After installation, call trainer commands directly, or use `uv run --no-sync`
from the trainer checkout. When updating dependencies, use `uv sync --locked
--inexact --preview-features extra-build-dependencies` to retain separately
installed preparation packages. A normal exact sync can remove those packages.

## Separate environments

Preparation and training may use separate environments or different machines.
Both need their own dependencies; preparation still needs access to the trainer
helper source described above. Pass the capture directory (images, masks,
COLMAP model, initialization NPZ, and optional timing JSON) between stages.
Optional backends with incompatible dependencies need their own environments.

See the [core README](README.md) for the complete pipeline and the
[trainer README](https://github.com/ASU-NEM-Labs/open4dgs-trigs) for training.
