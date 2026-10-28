"""CSV output denominators, distinct-string collisions, and CLI contracts."""

import csv
import json
import subprocess
import sys

import pytest

from bondaries.batch import batch
from bondaries.cli import main
from bondaries.registry import get_model, load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("chemberta", local_files_only=True)


def write_input(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "unrelated"])
        writer.writerows((value, "not retained") for value in values)


def test_batch_drops_and_distinct_smiles_collapses(tmp_path, tokenizer):
    source = tmp_path / "input.csv"
    values = ["BrC1=CC=CC=C1Cl", "BC1=CC=CC=C1C", "BC1=CC=CC=C1C", "CCO"]
    write_input(source, [value + "\n" for value in values])
    out = tmp_path / "results"
    result = batch(source, tokenizer, out=out)
    assert result["n"] == 4
    assert result["dropped_characters"] == 2
    assert result["molecule_dropped_fraction"] == 0.25
    assert result["dropped_character_fraction"] == 2 / sum(map(len, values))
    assert result["collapse_group_count"] == 1
    assert result["collapse_distinct_smiles"] == 2
    assert result["collapse_rows"] == 3
    with (out / "molecules.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["smiles"] for row in rows] == values
    assert all(row["n"] == "4" for row in rows)
    assert "unrelated" not in rows[0]
    assert len(list(csv.DictReader((out / "crimes.csv").open()))) == 4
    groups = list(csv.DictReader((out / "collapse_groups.csv").open()))
    assert sorted(int(row["occurrences"]) for row in groups) == [1, 2]


def test_repeated_identical_smiles_are_not_collapses(tmp_path, tokenizer):
    source = tmp_path / "input.csv"
    write_input(source, ["CCO", "CCO"])
    out = tmp_path / "out"
    first = batch(source, tokenizer, out=out)
    assert first["collapse_group_count"] == 0
    assert batch(source, tokenizer, out=out) == first


def test_empty_csv_and_missing_column(tmp_path, tokenizer):
    source = tmp_path / "input.csv"
    write_input(source, [])
    result = batch(source, tokenizer, out=tmp_path / "out")
    assert result["n"] == 0
    assert result["dropped_character_fraction"] is None
    assert result["mean_tokens_per_molecule"] is None
    with pytest.raises(ValueError, match="no column"):
        batch(source, tokenizer, col="absent", out=tmp_path / "bad")


def test_cli_pinned_revision_and_outputs(tmp_path, capsys):
    source = tmp_path / "input.csv"
    write_input(source, ["BrC1=CC=CC=C1Cl"])
    model = get_model("chemberta")
    assert main(["batch", str(source), "--tokenizer", model["repo"], "--revision", model["sha"],
                 "--local-files-only", "--out", str(tmp_path / "out")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["revision"] == model["sha"]
    assert result["n"] == 1
    assert result["dropped_characters"] == 2


def test_module_help():
    result = subprocess.run([sys.executable, "-m", "bondaries", "--help"],
                            capture_output=True, text=True, check=True)
    assert "batch" in result.stdout


@pytest.mark.parametrize("filename", [
    "molecules.csv", "crimes.csv", "collapse_groups.csv", "summary.json",
])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_output_collision_preserves_input_before_analysis(tmp_path, filename, alias):
    if alias == "direct":
        source, out = tmp_path / filename, tmp_path
    else:
        source, out = tmp_path / "input.csv", tmp_path / "results"
        out.mkdir()
    write_input(source, ["CCO", "BrC"])
    original = source.read_bytes()
    if alias == "symlink":
        (out / filename).symlink_to(source)
    elif alias == "hardlink":
        (out / filename).hardlink_to(source)
    existing = set(out.iterdir())
    # An unusable tokenizer ensures the path check runs before tokenization.
    with pytest.raises(ValueError, match="would overwrite the input CSV"):
        batch(source, object(), out=out)
    assert source.read_bytes() == original
    assert set(out.iterdir()) == existing


def test_cli_collision_is_a_clean_error(tmp_path, monkeypatch, capsys):
    source = tmp_path / "molecules.csv"
    write_input(source, ["CCO"])
    original = source.read_bytes()
    monkeypatch.setattr("bondaries.registry.load_tokenizer", lambda *args, **kwargs: object())
    with pytest.raises(SystemExit) as error:
        main(["batch", str(source), "--tokenizer", "chemberta", "--out", str(tmp_path)])
    assert error.value.code == 2
    assert "would overwrite the input CSV" in capsys.readouterr().err
    assert source.read_bytes() == original
