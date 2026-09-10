from pathlib import Path
from unittest.mock import patch

from msprobe.logic_buffer_trace import logic_monkeypatch


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(logic_monkeypatch, "_INSTALLED", True)
    with patch.object(logic_monkeypatch, "_load_helper") as load_helper:
        logic_monkeypatch.install()
    load_helper.assert_not_called()


def test_auto_viz_finds_non_model_named_directory(tmp_path, monkeypatch):
    graph_dir = tmp_path / "torch_compile_debug" / "run" / "BERT_pytorch__0_forward_1.0"
    graph_dir.mkdir(parents=True)
    (graph_dir / "output_code.py").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INDUCTOR_LOGIC_BUFFER_AUTO_VIZ", "1")

    with patch("subprocess.run") as run:
        logic_monkeypatch.maybe_auto_viz()

    command = run.call_args.args[0]
    assert Path(command[2]) == graph_dir
