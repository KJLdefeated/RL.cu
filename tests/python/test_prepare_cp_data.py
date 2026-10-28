"""Offline tests for the CodeContests -> rollout JSONL data contract."""
import copy
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_cp_data as cp


@pytest.fixture
def problem():
    return {
        "name": "sum",
        "description": "Add two integers. 整数求和",
        "difficulty": 2,
        "public_tests": {"input": ["1 2\n"], "output": ["3\n"]},
        "private_tests": {"input": ["3 4\n"], "output": ["7\n"]},
        "generated_tests": {"input": ["5 6\n"], "output": ["11\n"]},
    }


@pytest.mark.parametrize("field", [None, {}, {"input": [], "output": []}])
def test_collect_pairs_empty(field):
    assert cp._collect_pairs(field, None) == []


@pytest.mark.parametrize("cap, count", [(None, 2), (0, 0), (1, 1), (10, 2)])
def test_collect_pairs_caps_and_unequal_lengths(cap, count):
    field = {"input": ["a", "b", "unpaired"], "output": ["A", "B"]}
    assert cp._collect_pairs(field, cap) == [
        {"input": "a", "expected": "A"}, {"input": "b", "expected": "B"}
    ][:count]


@pytest.mark.parametrize("missing", ["name", "public_tests", "hidden_tests"])
def test_extract_rejects_unusable_problem(problem, missing):
    if missing == "hidden_tests":
        problem.pop("private_tests")
        problem.pop("generated_tests")
    else:
        problem.pop(missing)
    assert cp.extract_problem(problem, 10) is None


@pytest.mark.parametrize("cap, expected", [(None, ["7\n", "11\n"]),
                                           (1, ["7\n"]),
                                           (10, ["7\n", "11\n"])])
def test_extract_prioritizes_private_tests_and_preserves_input(problem, cap, expected):
    original = copy.deepcopy(problem)
    result = cp.extract_problem(problem, cap)
    assert result["id"] == "sum"
    assert result["description"] == problem["description"]
    assert result["public_tests"] == [{"input": "1 2\n", "expected": "3\n"}]
    assert [pair["expected"] for pair in result["hidden_tests"]] == expected
    assert problem == original


def test_extract_allows_missing_description_and_generated_only(problem):
    problem.pop("description")
    problem.pop("private_tests")
    result = cp.extract_problem(problem, 10)
    assert result["description"] == ""
    assert result["hidden_tests"] == [{"input": "5 6\n", "expected": "11\n"}]


def test_list_shards_filters_split_and_sorts(monkeypatch):
    def list_files(repo_id, repo_type):
        assert (repo_id, repo_type) == ("local/test", "dataset")
        return ["README.md", "data/train-01.parquet", "data/test-02.parquet",
                "data/test-01.parquet", "other/test-00.parquet", "data/test-01.json"]

    monkeypatch.setattr(cp, "list_repo_files", list_files)
    assert cp.list_shards("local/test", "test") == [
        "data/test-01.parquet", "data/test-02.parquet"
    ]


def test_download_shards_forwards_dataset_and_cache(monkeypatch):
    calls = []

    def download(repo_id, path, **kwargs):
        calls.append((repo_id, path, kwargs))
        return "/cache/" + path

    monkeypatch.setattr(cp, "hf_hub_download", download)
    assert cp.download_shards("local/test", ["a.parquet", "b.parquet"], "/cache") == [
        "/cache/a.parquet", "/cache/b.parquet"
    ]
    assert calls == [("local/test", path, {"repo_type": "dataset", "cache_dir": "/cache"})
                     for path in ["a.parquet", "b.parquet"]]


@pytest.mark.parametrize("extra, expected_ids", [
    ([], ["sum", "hard"]),
    (["--max-samples", "1"], ["sum"]),
    (["--difficulty-min", "3"], ["hard"]),
    (["--difficulty-max", "2"], ["sum"]),
    (["--min-public", "2"], []),
])
def test_cli_writes_rollout_jsonl_offline(tmp_path, monkeypatch, problem, extra, expected_ids):
    hard = copy.deepcopy(problem)
    hard.update(name="hard", difficulty=5)
    invalid = copy.deepcopy(problem)
    invalid["name"] = ""
    shard = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist([invalid, problem, hard]), shard)
    output = tmp_path / "nested" / "problems.jsonl"
    seen_splits = []

    def list_shards(repo_id, split):
        seen_splits.append(split)
        return ["data/valid-00.parquet"]

    monkeypatch.setattr(cp, "list_shards", list_shards)
    monkeypatch.setattr(cp, "download_shards", lambda *args: [str(shard)])
    monkeypatch.setattr(sys, "argv", ["prepare_cp_data.py", "--output", str(output),
                                      "--split", "validation", "--max-hidden", "1", *extra])
    cp.main()
    assert seen_splits == ["valid"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == expected_ids
    for row in rows:
        assert set(row) == {"id", "description", "public_tests", "hidden_tests"}
        assert row["description"] == problem["description"]
        assert row["hidden_tests"] == [{"input": "3 4\n", "expected": "7\n"}]


def test_cli_missing_shards_returns_error(tmp_path, monkeypatch, capsys):
    output = tmp_path / "problems.jsonl"
    monkeypatch.setattr(cp, "list_shards", lambda *args: [])
    monkeypatch.setattr(sys, "argv", ["prepare_cp_data.py", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        cp.main()
    assert exc.value.code == 2
    assert "no test shards" in capsys.readouterr().err
    assert not output.exists()
