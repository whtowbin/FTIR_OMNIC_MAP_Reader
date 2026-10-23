"""check_zarr_store refuses Zarr metadata that could run code on read (pickle codecs, object arrays, links)."""

import json

import pytest

from ftir_map import read_unpacked
from ftir_map.zarr_safety import UnsafeZarrStore, check_zarr_store

from .test_exports import grid_map


def v2_store(tmp_path, **zarray):
    store = tmp_path / "store.zarr"
    (store / "arr").mkdir(parents=True)
    meta = {"zarr_format": 2, "dtype": "<f4", "compressor": {"id": "blosc"}, "filters": None} | zarray
    (store / "arr" / ".zarray").write_text(json.dumps(meta))
    return store


def v3_store(tmp_path, data_type="float32", codecs=None):
    store = tmp_path / "store.zarr"
    (store / "arr").mkdir(parents=True)
    codecs = codecs if codecs is not None else [{"name": "bytes"}, {"name": "zstd"}]
    meta = {"zarr_format": 3, "node_type": "array", "data_type": data_type, "codecs": codecs}
    (store / "arr" / "zarr.json").write_text(json.dumps(meta))
    return store


def test_a_store_written_by_unpack_passes(tmp_path):
    grid_map(tmp_path).unpack(tmp_path / "u")
    check_zarr_store(tmp_path / "u" / "spectra.zarr")


@pytest.mark.parametrize(
    "zarray",
    [
        {"dtype": "|O", "filters": [{"id": "pickle"}]},
        {"dtype": "|O", "filters": None},
        {"dtype": "<f4", "filters": [{"id": "pickle"}]},
        {"dtype": "<f4", "compressor": {"id": "pickle"}},
        {"dtype": [["a", "<f4"], ["b", "|O"]]},
    ],
)
def test_v2_unsafe_metadata_is_refused(tmp_path, zarray):
    with pytest.raises(UnsafeZarrStore):
        check_zarr_store(v2_store(tmp_path, **zarray))


def test_v2_utf8_strings_are_allowed(tmp_path):
    check_zarr_store(v2_store(tmp_path, dtype="|O", filters=[{"id": "vlen-utf8"}]))
    check_zarr_store(v2_store(tmp_path / "b", dtype="<U15"))


@pytest.mark.parametrize(
    ("data_type", "codecs"),
    [
        ("float32", [{"name": "bytes"}, {"name": "numcodecs.pickle"}]),
        ("object", [{"name": "bytes"}]),
        ({"name": "some_custom_type"}, [{"name": "bytes"}]),
        ("float32", [{"name": "sharding_indexed", "configuration": {"codecs": [{"name": "numcodecs.pickle"}]}}]),
    ],
)
def test_v3_unsafe_metadata_is_refused(tmp_path, data_type, codecs):
    with pytest.raises(UnsafeZarrStore):
        check_zarr_store(v3_store(tmp_path, data_type, codecs))


def test_v3_safe_metadata_is_allowed(tmp_path):
    check_zarr_store(v3_store(tmp_path, {"name": "fixed_length_utf32", "configuration": {"length_bytes": 60}}))


def test_links_inside_a_store_are_refused(tmp_path):
    store = v3_store(tmp_path)
    (tmp_path / "secret").write_text("x")
    (store / "arr" / "c").symlink_to(tmp_path / "secret")
    with pytest.raises(UnsafeZarrStore, match="symbolic link"):
        check_zarr_store(store)


def test_read_unpacked_refuses_a_store_tampered_to_use_pickle(tmp_path):
    folder = tmp_path / "u"
    grid_map(tmp_path).unpack(folder)
    spectra = folder / "spectra.zarr" / "spectra"
    if (spectra / ".zarray").exists():
        meta = json.loads((spectra / ".zarray").read_text())
        meta["filters"] = [{"id": "pickle"}]
        (spectra / ".zarray").write_text(json.dumps(meta))
    else:
        meta = json.loads((spectra / "zarr.json").read_text())
        meta["codecs"].append({"name": "numcodecs.pickle"})
        (spectra / "zarr.json").write_text(json.dumps(meta))
    with pytest.raises(UnsafeZarrStore):
        read_unpacked(folder)


@pytest.mark.parametrize("name", [".ZARRAY", ".Zarray", "ZARR.JSON", "Zarr.json"])
def test_metadata_names_spelled_in_another_case_are_refused(tmp_path, name):
    store = tmp_path / "store.zarr"
    (store / "arr").mkdir(parents=True)
    meta = {"zarr_format": 2, "dtype": "|O", "compressor": None, "filters": [{"id": "pickle"}]}
    (store / "arr" / name).write_text(json.dumps(meta))  # on macOS zarr would read this as .zarray / zarr.json
    with pytest.raises(UnsafeZarrStore, match="not spelled"):
        check_zarr_store(store)
