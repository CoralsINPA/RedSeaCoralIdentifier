import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import coralid
from coralid import colab
from coralid.colab import (
    BOOTSTRAP_CELL,
    activate_editable_install,
    check_data_root,
    git_steps,
    installed_versions,
    parse_extras,
    pip_step,
    replaced_packages,
    restart_message,
    setup,
)


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "CoralsMasking"
    (root / "results_all").mkdir(parents=True)
    return root


# ------------------------------------------------------------------------- the data-root checks


def test_a_missing_data_root_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="DATA_ROOT not found") as err:
        check_data_root(tmp_path / "nowhere")
    assert "add a shortcut to it in My Drive first" in str(err.value)


def test_a_data_root_without_results_all_means_the_wrong_account(tmp_path):
    """Commit 9f6f9ee: a mount with another Google account gives an empty folder at that path."""
    (tmp_path / "CoralsMasking").mkdir()
    with pytest.raises(FileNotFoundError, match="has no results_all/ folder") as err:
        check_data_root(tmp_path / "CoralsMasking")
    text = str(err.value)
    assert "different Google account" in text and "Disconnect and delete runtime" in text


def test_the_checks_pass_for_the_real_layout(data_root):
    check_data_root(data_root)


# ------------------------------------------------------------------------- command lists


def test_git_steps_update_a_clone_or_clone_with_submodules():
    update = git_steps("/c/repo", "mybranch", "https://example.org/r", cloned=True)
    assert update == [
        ["git", "-C", "/c/repo", "fetch", "origin", "mybranch"],
        ["git", "-C", "/c/repo", "checkout", "mybranch"],
        ["git", "-C", "/c/repo", "pull", "--ff-only", "origin", "mybranch"],
        ["git", "-C", "/c/repo", "submodule", "update", "--init"],
    ]
    fresh = git_steps("/c/repo", "mybranch", "https://example.org/r", cloned=False)
    assert fresh == [
        [
            "git",
            "clone",
            "--recurse-submodules",
            "--branch",
            "mybranch",
            "https://example.org/r",
            "/c/repo",
        ]
    ]


def test_pip_step_installs_the_clone_editable_with_extras():
    step = pip_step("/c/repo", "clip,viz", python="/usr/bin/python3")
    assert step == ["/usr/bin/python3", "-m", "pip", "install", "-q", "-e", "/c/repo[clip,viz]"]
    assert pip_step("/c/repo", "")[-1] == "/c/repo"
    assert pip_step("/c/repo", "viz")[0] == sys.executable


def test_extras_are_normalized_and_checked():
    assert parse_extras("clip, viz") == "clip,viz"
    assert parse_extras(["segment", "viz"]) == "segment,viz"
    assert parse_extras("") == ""
    with pytest.raises(ValueError, match="Unknown extras"):
        parse_extras("clip,vizz")


def test_every_extra_exists_in_pyproject():
    import tomllib

    root = Path(coralid.__file__).resolve().parents[2]
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert set(colab.KNOWN_EXTRAS) == set(declared)


# ------------------------------------------------------------------------- versions and restarts


def _dist(name, version):
    return SimpleNamespace(metadata={"Name": name}, version=version)


def test_installed_versions_normalizes_names():
    found = installed_versions([_dist("open_clip_torch", "3.3.0"), _dist("Pillow", "11.0")])
    assert found == {"open-clip-torch": "3.3.0", "pillow": "11.0"}
    assert installed_versions()  # the real environment has packages


def test_only_watched_packages_that_changed_need_a_restart():
    before = {"numpy": "2.0", "pandas": "2.2", "torch": "2.5", "tqdm": "4.0", "pillow": "10"}
    after = {"numpy": "2.1", "pandas": "2.2", "torch": "2.5", "tqdm": "4.5", "matplotlib": "3.9"}
    assert replaced_packages(before, after) == {"numpy": ("2.0", "2.1"), "pillow": ("10", None)}
    assert replaced_packages(before, before) == {}
    # a package pip installed for the first time cannot have been loaded
    assert replaced_packages({}, {"torch": "2.5"}) == {}


def test_the_restart_message_names_the_packages():
    text = restart_message({"numpy": ("2.0", "2.1"), "pillow": ("10", None)})
    assert "numpy 2.0 -> 2.1, pillow 10 -> None" in text
    assert "RESTART THE RUNTIME" in text and "run this cell again" in text


def test_the_editable_install_is_made_importable(tmp_path, monkeypatch):
    site_dir, source = tmp_path / "site", tmp_path / "src_here"
    site_dir.mkdir()
    source.mkdir()
    pth = site_dir / "__editable__.coralid-0.1.0.pth"
    pth.write_text(f"{source}\n")
    (site_dir / "unrelated.pth").write_text("/nonexistent\n")
    seen = []
    read = activate_editable_install(None, [str(site_dir)], lambda d, n, k: seen.append((d, n)))
    assert read == [str(pth)] and seen == [(str(site_dir), pth.name)]  # only coralid's file
    monkeypatch.setattr(sys, "path", list(sys.path))
    activate_editable_install(None, [str(site_dir)])  # the real site.addpackage reads the file
    assert str(source) in sys.path


def test_without_an_install_the_clones_src_folder_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    activate_editable_install(tmp_path / "repo", [])
    assert sys.path[0] == str(tmp_path / "repo" / "src")


# ------------------------------------------------------------------------- setup()


class Recorder:
    """A fake `subprocess.run` that records the commands."""

    def __init__(self):
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append((list(command), kwargs))
        return SimpleNamespace(stdout="abc1234\n", returncode=0)


@pytest.fixture
def env_cleanup(monkeypatch):
    monkeypatch.delenv(colab.DATA_ROOT_ENV, raising=False)


def test_setup_runs_the_notebooks_setup_cell(tmp_path, data_root, env_cleanup):
    run, lines = Recorder(), []
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    out = setup(
        "clip,viz",
        data_root=str(data_root),
        branch="b",
        repo_url="https://example.org/r",
        repo_dir=str(repo),
        mount=False,
        runner=run,
        versions=lambda: {"numpy": "2.0", "torch": "2.5"},
        echo=lines.append,
    )
    commands = [c for c, _ in run.commands]
    assert commands[:4] == git_steps(str(repo), "b", "https://example.org/r", cloned=True)
    assert commands[4] == pip_step(str(repo), "clip,viz")
    assert commands[5] == ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]
    assert all(kw == {"check": True} for _, kw in run.commands[:5])
    assert os.environ["CORALID_DATA_ROOT"] == str(data_root)
    assert out.commit == "abc1234" and out.data_root == data_root and out.extras == "clip,viz"
    assert out.config_path() == repo / "configs" / "iou60_sta55_pts32.yaml"
    assert any("Setup done. No restart needed." in line for line in lines)
    assert any(
        f"coralid {coralid.__version__} at commit abc1234 (branch b)" in line for line in lines
    )


def test_setup_clones_when_there_is_no_clone_yet(tmp_path, data_root, env_cleanup):
    run = Recorder()
    setup(
        "viz",
        data_root=str(data_root),
        repo_dir=str(tmp_path / "fresh"),
        mount=False,
        runner=run,
        versions=lambda: {},
        echo=lambda line: None,
    )
    assert run.commands[0][0][:3] == ["git", "clone", "--recurse-submodules"]


def test_a_wrong_data_root_stops_before_git_and_pip(tmp_path, env_cleanup):
    run = Recorder()
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="results_all"):
        setup(data_root=str(tmp_path / "empty"), mount=False, runner=run, echo=lambda line: None)
    assert run.commands == []


def test_unknown_extras_stop_before_anything_happens(tmp_path, data_root, env_cleanup):
    run = Recorder()
    with pytest.raises(ValueError, match="Unknown extras"):
        setup("nope", data_root=str(data_root), mount=False, runner=run)
    assert run.commands == [] and colab.DATA_ROOT_ENV not in os.environ


def test_setup_demands_a_restart_when_pip_replaced_a_loaded_package(
    tmp_path, data_root, env_cleanup
):
    snapshots = iter([{"numpy": "2.0"}, {"numpy": "2.1"}])
    with pytest.raises(RuntimeError, match="RESTART THE RUNTIME") as err:
        setup(
            data_root=str(data_root),
            repo_dir=str(tmp_path / "r"),
            mount=False,
            runner=Recorder(),
            versions=lambda: next(snapshots),
            echo=lambda line: None,
        )
    assert "numpy 2.0 -> 2.1" in str(err.value)


def test_setup_mounts_drive_first(tmp_path, data_root, env_cleanup, monkeypatch):
    order = []
    monkeypatch.setattr(colab, "mount_drive", lambda: order.append("mount"))
    run = Recorder()

    def recording(command, **kwargs):
        order.append(command[0])
        return run(command, **kwargs)

    setup(
        data_root=str(data_root),
        repo_dir=str(tmp_path / "r"),
        runner=recording,
        versions=lambda: {},
        echo=lambda line: None,
    )
    assert order[0] == "mount" and order[1] == "git"


def test_setup_warns_when_the_two_opencv_builds_differ(tmp_path, data_root, env_cleanup):
    lines = []
    snapshots = iter(
        [{}, {"opencv-python": "4.12.0", "opencv-python-headless": "4.11.0"}],
    )
    setup(
        data_root=str(data_root),
        repo_dir=str(tmp_path / "r"),
        mount=False,
        runner=Recorder(),
        versions=lambda: next(snapshots),
        echo=lines.append,
    )
    assert any("opencv-python and opencv-python-headless differ" in line for line in lines)


# ------------------------------------------------------------------------- the first cell


def test_the_bootstrap_cell_clones_then_hands_over_to_setup(tmp_path, monkeypatch):
    """Run the documented first cell with fakes: it must clone, find coralid.colab, call setup."""
    repo = tmp_path / "RedSeaCoralIdentifier"
    cell = BOOTSTRAP_CELL.replace('"/content/RedSeaCoralIdentifier"', repr(str(repo)))
    compile(cell, "<bootstrap>", "exec")  # valid Python
    commands, calls = [], []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append((command, kw)))
    monkeypatch.setattr(colab, "setup", lambda *a, **kw: calls.append((a, kw)) or "ENV")
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"__name__": "__notebook__"}
    exec(cell, namespace)
    (clone, kwargs) = commands[0]
    assert clone[:4] == ["git", "clone", "--recurse-submodules", "--branch"]
    assert clone[-1] == str(repo) and kwargs == {"check": True}
    assert sys.path[0] == f"{repo}/src"
    ((args, kw),) = calls
    assert args == ("clip,viz",) and kw["repo_dir"] == str(repo)
    assert kw["data_root"] == namespace["DATA_ROOT"] == colab.DEFAULT_DATA_ROOT
    assert kw["branch"] == colab.DEFAULT_BRANCH and kw["repo_url"] == colab.DEFAULT_REPO_URL
    assert namespace["env"] == "ENV"
    # an existing clone is not cloned again
    (repo / ".git").mkdir(parents=True)
    commands.clear()
    exec(cell, {"__name__": "__notebook__"})
    assert commands == []


def test_the_bootstrap_needs_only_the_standard_library_to_import_colab():
    code = (
        "import sys, coralid.colab; "
        "heavy = {'torch', 'pandas', 'numpy', 'cv2', 'yaml', 'pyarrow'}; "
        "loaded = heavy & set(sys.modules); assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_defaults_match_the_notebooks_setup_cell():
    notebook = Path(coralid.__file__).resolve().parents[2] / "notebooks" / "03_zero_shot.ipynb"
    if not notebook.exists():
        pytest.skip("notebook 03 is not in this checkout")
    text = notebook.read_text()
    for value in (
        colab.DEFAULT_DATA_ROOT,
        colab.DEFAULT_BRANCH,
        colab.DEFAULT_REPO_URL,
        colab.DEFAULT_REPO_DIR,
    ):
        assert value in text
