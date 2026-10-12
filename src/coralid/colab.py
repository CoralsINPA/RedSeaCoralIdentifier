"""Colab: the only module that knows about Colab (Drive, the install, runtime restarts).

`setup()` is the setup cell of the notebooks in one call. In order, it:

1. mounts Drive and sets `CORALID_DATA_ROOT` (the data location every other module reads);
2. checks the data root: the folder must exist, **and** it must hold `results_all/`. The second
   check exists because Drive was once mounted with the wrong Google account, whose own folder at
   the same path was empty, so everything looked missing;
3. clones this repository with its submodules (CoralSCOP), or updates the clone, and runs
   `pip install -e` with the requested extras. A plain `pip install git+https://...` would miss
   `configs/` and the submodule;
4. makes the editable install importable in the running interpreter (an editable install adds a
   `.pth` file, which a running interpreter reads only at startup);
5. stops with "restart the runtime" if pip replaced a package this session may have loaded.

**Bootstrap.** `coralid.colab` cannot be imported before the repository is on disk, so the first
cell of a notebook is `BOOTSTRAP_CELL`: it clones the repository if needed, puts its `src/` folder
on `sys.path` (`coralid/__init__.py` and this module need only the standard library), and calls
`setup()`, which does everything else. Copy the cell, edit its settings block, and nothing more.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import site
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/colab notebooks/CoralsMasking"
DEFAULT_BRANCH = "claude/awesome-archimedes-kvh4c7"
DEFAULT_REPO_URL = "https://github.com/CoralsINPA/RedSeaCoralIdentifier"
DEFAULT_REPO_DIR = "/content/RedSeaCoralIdentifier"
DRIVE_MOUNT = "/content/drive"
DATA_ROOT_ENV = "CORALID_DATA_ROOT"  # the same variable as `coralid.config.DATA_ROOT_ENV`

KNOWN_EXTRAS = ("clip", "dino", "segment", "cluster", "probe", "viz", "dev")
# Packages that are usually already loaded in a Colab session: replacing one needs a restart.
WATCHED_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "pillow",
    "opencv-python",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "open-clip-torch",
    "transformers",
    "matplotlib",
)
REPORTED_PACKAGES = (
    "opencv-python",
    "opencv-python-headless",
    "torch",
    "open-clip-torch",
    "matplotlib",
)

BOOTSTRAP_CELL = """\
import os
import subprocess
import sys

# ================================ SETTINGS: the only place to edit ================================
DATA_ROOT = "/content/drive/MyDrive/colab notebooks/CoralsMasking"  # the Drive folder CoralsMasking
BRANCH = "claude/awesome-archimedes-kvh4c7"
REPO_URL = "https://github.com/CoralsINPA/RedSeaCoralIdentifier"
REPO_DIR = "/content/RedSeaCoralIdentifier"
EXTRAS = "clip,viz"
# ==================================================================================================

if not os.path.isdir(f"{REPO_DIR}/.git"):  # the package cannot install itself before it exists
    clone = ["git", "clone", "--recurse-submodules", "--branch", BRANCH, REPO_URL, REPO_DIR]
    subprocess.run(clone, check=True)
sys.path.insert(0, f"{REPO_DIR}/src")  # enough to import coralid.colab (standard library only)

from coralid.colab import setup  # noqa: E402

env = setup(EXTRAS, data_root=DATA_ROOT, branch=BRANCH, repo_url=REPO_URL, repo_dir=REPO_DIR)
"""


def check_data_root(data_root: str | os.PathLike[str]) -> None:
    """Raise `FileNotFoundError` unless `data_root` exists and holds `results_all/`."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"DATA_ROOT not found: {data_root}\n"
            "Edit DATA_ROOT above. If the folder was shared with you, "
            "add a shortcut to it in My Drive first."
        )
    if not os.path.isdir(os.path.join(data_root, "results_all")):
        raise FileNotFoundError(
            f"DATA_ROOT exists but has no results_all/ folder: {data_root}\n"
            "So this is not the shared data folder. Most likely Drive is mounted with a different "
            "Google account than the one the data is shared with: Runtime > Disconnect and delete "
            "runtime, run this cell again and choose that account. Otherwise fix DATA_ROOT."
        )


def parse_extras(extras: str | Sequence[str]) -> str:
    """`clip,viz` or `["clip", "viz"]` as the string pip wants; unknown names are an error."""
    names = [e.strip() for e in (extras.split(",") if isinstance(extras, str) else extras)]
    names = [n for n in names if n]
    unknown = [n for n in names if n not in KNOWN_EXTRAS]
    if unknown:
        raise ValueError(f"Unknown extras {unknown}; choose from {list(KNOWN_EXTRAS)}")
    return ",".join(names)


def git_steps(repo_dir: str, branch: str, repo_url: str, cloned: bool) -> list[list[str]]:
    """The git commands: update an existing clone, or clone with the submodules."""
    if cloned:
        return [
            ["git", "-C", repo_dir, "fetch", "origin", branch],
            ["git", "-C", repo_dir, "checkout", branch],
            ["git", "-C", repo_dir, "pull", "--ff-only", "origin", branch],
            ["git", "-C", repo_dir, "submodule", "update", "--init"],
        ]
    return [["git", "clone", "--recurse-submodules", "--branch", branch, repo_url, repo_dir]]


def pip_step(repo_dir: str, extras: str, python: str | None = None) -> list[str]:
    """`pip install -e <repo>[extras]` with the interpreter that runs the notebook."""
    target = f"{repo_dir}[{extras}]" if extras else repo_dir
    return [python or sys.executable, "-m", "pip", "install", "-q", "-e", target]


def installed_versions(
    distributions: Iterable[Any] | None = None,
) -> dict[str, str]:
    """Installed packages as `{normalized name: version}` (names lower case, `_` as `-`)."""
    found = metadata.distributions() if distributions is None else distributions
    return {d.metadata["Name"].lower().replace("_", "-"): d.version for d in found}


def replaced_packages(
    before: dict[str, str],
    after: dict[str, str],
    watch: Iterable[str] = WATCHED_PACKAGES,
) -> dict[str, tuple[str, str | None]]:
    """Watched packages whose version changed: `{name: (before, after)}`; `after` may be None."""
    return {
        k: (before[k], after.get(k)) for k in watch if k in before and before[k] != after.get(k)
    }


def restart_message(replaced: dict[str, tuple[str, str | None]]) -> str:
    """The error text that tells the user to restart the runtime."""
    changes = ", ".join(f"{k} {a} -> {b}" for k, (a, b) in replaced.items())
    return (
        "pip replaced packages that this Python session may already have loaded: "
        f"{changes}\n\n>>> RESTART THE RUNTIME now (Runtime > Restart session), "
        "then run this cell again and continue. <<<"
    )


def activate_editable_install(
    repo_dir: str | os.PathLike[str] | None = None,
    site_dirs: Sequence[str] | None = None,
    add_package: Callable[[str, str, Any], Any] = site.addpackage,
) -> list[str]:
    """Make a fresh editable install importable now; return the `.pth` files that were read.

    If `coralid` is still not importable, the clone's `src/` folder is put on `sys.path`.
    """
    if site_dirs is None:
        site_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    activated = []
    for site_dir in site_dirs:
        for pth in glob.glob(os.path.join(site_dir, "__editable__*coralid*.pth")):
            add_package(site_dir, os.path.basename(pth), None)
            activated.append(pth)
    importlib.invalidate_caches()
    if repo_dir is not None and importlib.util.find_spec("coralid") is None:
        sys.path.insert(0, os.path.join(os.fspath(repo_dir), "src"))
    return activated


def mount_drive(mount_point: str = DRIVE_MOUNT) -> None:
    """Mount Google Drive (this import exists only inside Colab)."""
    from google.colab import drive

    drive.mount(mount_point)


@dataclass(frozen=True)
class ColabSetup:
    """What `setup()` returns."""

    repo_dir: Path
    data_root: Path
    branch: str
    commit: str
    extras: str

    def config_path(self, name: str = "iou60_sta55_pts32.yaml") -> Path:
        """A run config of the cloned repository (`configs/<name>`)."""
        return self.repo_dir / "configs" / name


def setup(
    extras: str | Sequence[str] = "clip,viz",
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    branch: str = DEFAULT_BRANCH,
    repo_url: str = DEFAULT_REPO_URL,
    repo_dir: str = DEFAULT_REPO_DIR,
    mount: bool = True,
    runner: Callable[..., Any] = subprocess.run,
    versions: Callable[[], dict[str, str]] = installed_versions,
    echo: Callable[[str], None] = print,  # a notebook cell shows these lines to the user
) -> ColabSetup:
    """Mount Drive, set the data root, install the package and say whether to restart.

    Raises `FileNotFoundError` (before touching git or pip) if the data root is wrong, and
    `RuntimeError` if pip replaced a package that is probably loaded; restart the runtime then
    and run the cell again. `mount`, `runner`, `versions` and `echo` are for tests.
    """
    extras_text = parse_extras(extras)
    if mount:
        mount_drive()
    os.environ[DATA_ROOT_ENV] = data_root  # coralid reads the data location from here
    check_data_root(data_root)

    before = versions()
    cloned = os.path.isdir(os.path.join(repo_dir, ".git"))
    for step in [*git_steps(repo_dir, branch, repo_url, cloned), pip_step(repo_dir, extras_text)]:
        echo("$ " + " ".join(step))
        runner(step, check=True)
    activate_editable_install(repo_dir)

    after = versions()
    replaced = replaced_packages(before, after)
    echo("\nInstalled versions (pip metadata):")
    for name in REPORTED_PACKAGES:
        echo(f"  {name:24} {after.get(name)}")
    cv, cv_headless = after.get("opencv-python"), after.get("opencv-python-headless")
    if cv and cv_headless and cv != cv_headless:
        echo("  WARNING: opencv-python and opencv-python-headless differ in version;")
        echo("  they share one cv2 folder.")

    coralid = importlib.import_module("coralid")  # importable only after the install above
    head = runner(
        ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    commit = (head.stdout or "").strip()
    echo(f"\ncoralid {coralid.__version__} at commit {commit} (branch {branch})")
    if replaced:
        raise RuntimeError(restart_message(replaced))
    echo("\nSetup done. No restart needed.")
    return ColabSetup(Path(repo_dir), Path(data_root), branch, commit, extras_text)
