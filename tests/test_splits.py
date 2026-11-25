"""Проверки зафиксированного в git разбиения (не требуют самих снимков)."""

import csv
import json

from polypseg.config import file_sha256

SPLITS = ("train", "val", "test")


def read(root, split):
    return (root / "data" / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()


def test_splits_disjoint_and_complete(root):
    splits = {s: read(root, s) for s in SPLITS}
    assert [len(splits[s]) for s in SPLITS] == [800, 100, 100]
    all_ids = set().union(*map(set, splits.values()))
    assert len(all_ids) == 1000
    for s in SPLITS:
        assert len(set(splits[s])) == len(splits[s])


def test_duplicates_in_same_split(root):
    where = {i: s for s in SPLITS for i in read(root, s)}
    with (root / "data" / "splits" / "duplicates.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert where[row["id_a"]] == where[row["id_b"]]


def test_manifest_matches_files(root):
    manifest = json.loads((root / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    for s in SPLITS:
        assert manifest["files_sha256"][s] == file_sha256(root / "data" / "splits" / f"{s}.txt")
