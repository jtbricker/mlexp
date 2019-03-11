"""Tests for `mlexp` package."""
import pytest
from mlexp import mlexp


def test_mirror(capsys):
    """Correct reverse name prints"""
    mlexp.mirror("Justin")
    captured = capsys.readouterr()
    assert "nitsuJ" in captured.out