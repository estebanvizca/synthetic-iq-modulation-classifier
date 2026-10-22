"""Tests for the CLI demo and figure generation."""

from __future__ import annotations

from pathlib import Path

import pytest

import demo


def test_demo_generates_prediction_and_local_figure(
    tmp_path: Path, trained_model: Path, capsys
) -> None:
    output = tmp_path / "figures" / "demo.png"
    exit_code = demo.main(
        [
            "--snr",
            "20",
            "--modulation",
            "qpsk",
            "--seed",
            "2026",
            "--output",
            str(output),
            "--model",
            str(trained_model),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "True modulation (simulator): qpsk" in captured.out
    assert "Prediction:" in captured.out
    assert "probabilities:" in captured.out
    assert "Reminder: this is a simulator result" in captured.out
    assert output.exists()
    assert output.stat().st_size > 10_000


def test_demo_random_modulation_is_reproducible(
    tmp_path: Path, trained_model: Path, capsys
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    args = [
        "--snr",
        "10",
        "--seed",
        "44",
        "--output",
        str(first),
        "--model",
        str(trained_model),
    ]
    assert demo.main(args) == 0
    first_text = capsys.readouterr().out
    args[args.index(str(first))] = str(second)
    assert demo.main(args) == 0
    second_text = capsys.readouterr().out
    first_name = first_text.split("True modulation (simulator): ", 1)[1].splitlines()[0]
    second_name = second_text.split("True modulation (simulator): ", 1)[1].splitlines()[0]
    assert first_name == second_name
    assert first.exists() and second.exists()


def test_demo_rejects_invalid_input(tmp_path: Path, trained_model: Path) -> None:
    with pytest.raises(SystemExit):
        demo.main(
            [
                "--snr",
                "nan",
                "--output",
                str(tmp_path / "bad.png"),
                "--model",
                str(trained_model),
            ]
        )
