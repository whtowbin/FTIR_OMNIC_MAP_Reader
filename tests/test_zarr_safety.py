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


@pytest.mark.parametrize(
    "codecs",
    [
        ["sharding_indexed"],
        [{"name": "sharding_indexed", "configuration": None}],
        [{"name": "sharding_indexed", "configuration": {"codecs": {"name": "bytes"}}}],
        [{"name": "sharding_indexed", "configuration": {"codecs": [{"name": "bytes"}], "index_codecs": "crc32c"}}],
        {"name": "bytes"},
        [["bytes"]],
        [{"name": ["bytes"]}],
    ],
)
def test_malformed_codec_structures_raise_unsafe_store(tmp_path, codecs):
    with pytest.raises(UnsafeZarrStore):
        check_zarr_store(v3_store(tmp_path, codecs=codecs))


def test_a_well_formed_sharding_codec_is_allowed(tmp_path):
    config = {"codecs": [{"name": "bytes"}], "index_codecs": [{"name": "bytes"}, {"name": "crc32c"}]}
    check_zarr_store(v3_store(tmp_path, codecs=[{"name": "sharding_indexed", "configuration": config}]))


def test_consolidated_metadata_is_never_used(tmp_path):
    """check_zarr_store reads the per-array metadata, so the store is opened with consolidated=False."""
    folder = tmp_path / "u"
    grid_map(tmp_path).unpack(folder)
    store = folder / "spectra.zarr"
    if (store / "zarr.json").exists():  # zarr 3 (Python 3.11 and later): consolidated copy in the root zarr.json
        root_meta = store / "zarr.json"
        content = json.loads(root_meta.read_text())
        content["consolidated_metadata"]["metadata"]["spectra"]["codecs"].append({"name": "numcodecs.pickle"})
    else:  # zarr 2 (Python 3.10): consolidated copy in .zmetadata
        root_meta = store / ".zmetadata"
        content = json.loads(root_meta.read_text())
        content["metadata"]["spectra/.zarray"]["filters"] = [{"id": "pickle"}]
    root_meta.write_text(json.dumps(content))
    check_zarr_store(folder / "spectra.zarr")  # the per-array metadata is still safe
    assert read_unpacked(folder).spectra.shape[0] == 6  # tampered consolidated copy is ignored


# ------------------------------------------------------------------ hand-made stores, one per refusal
def assert_refused(store, message):
    """check_zarr_store refuses `store` with a message that says why and names the offending path."""
    with pytest.raises(UnsafeZarrStore, match=message) as caught:
        check_zarr_store(store)
    assert str(store) in str(caught.value)


# Each store is a folder with one array's metadata file, as a stranger could ship it. The module docstring says what
# is allowed: numeric, boolean, fixed-width string and UTF-8 variable-length string arrays, standard compressors.
@pytest.mark.parametrize(
    "dtype",
    ["<c8", "<c16", "|V8", "<M8[ns]", "<m8[s]", [["a", "<f4"], ["b", "<i4"]], "not-a-dtype"],
    ids=["complex64", "complex128", "void", "datetime", "timedelta", "structured numeric", "invalid"],
)
def test_v2_dtypes_outside_the_allowed_kinds_are_refused(tmp_path, dtype):
    assert_refused(v2_store(tmp_path, dtype=dtype), "data type .* is not allowed")


def test_v2_unparsable_dtype_is_named_in_the_message(tmp_path):
    assert_refused(v2_store(tmp_path, dtype="not-a-dtype"), "data type 'not-a-dtype' is not allowed")


@pytest.mark.parametrize("dtype", ["|b1", "<i1", "<i8", "<u2", "<f2", "<f8", "<U20", "|S8"])
def test_v2_numeric_boolean_and_fixed_width_string_dtypes_are_allowed(tmp_path, dtype):
    check_zarr_store(v2_store(tmp_path, dtype=dtype))


@pytest.mark.parametrize(
    "filters",
    [
        None,
        [],
        [{"id": "pickle"}],
        [{"id": "vlen-utf8"}, {"id": "pickle"}],
        [{"id": "pickle"}, {"id": "vlen-utf8"}],
        [{"id": "vlen-bytes"}],
        ["pickle"],
        [{"id": "vlen-utf8"}, {"id": "vlen-utf8"}],
    ],
    ids=["none", "empty", "pickle", "utf8 then pickle", "pickle then utf8", "vlen-bytes", "pickle by name", "twice"],
)
def test_v2_object_arrays_need_exactly_the_utf8_filter(tmp_path, filters):
    assert_refused(v2_store(tmp_path, dtype="|O", filters=filters), "object data type with filters")


def test_v2_filters_may_be_given_by_name(tmp_path):
    check_zarr_store(v2_store(tmp_path, dtype="|O", filters=["vlen-utf8"]))
    assert_refused(
        v2_store(tmp_path / "b", dtype="<f4", filters=["delta"]), "with filters"
    )  # a numeric array needs none


@pytest.mark.parametrize("compressor", ["blosc", ["blosc"], 5, {"name": "blosc"}, {"id": "pickle"}, {"id": None}])
def test_v2_compressor_must_be_a_dict_with_a_standard_id(tmp_path, compressor):
    assert_refused(v2_store(tmp_path, compressor=compressor), "compressor .* is not allowed")


@pytest.mark.parametrize(
    "compressor", [None, *({"id": name} for name in ("blosc", "zstd", "gzip", "zlib", "lz4", "bz2", "lzma"))]
)
def test_v2_standard_compressors_are_allowed(tmp_path, compressor):
    check_zarr_store(v2_store(tmp_path, compressor=compressor))


@pytest.mark.parametrize(
    ("name", "content"),
    [(".zarray", b"{not json"), ("zarr.json", b""), (".zarray", b"\xff\xfe{")],
    ids=["invalid JSON", "empty file", "not UTF-8"],
)
def test_unreadable_metadata_is_refused(tmp_path, name, content):
    store = tmp_path / "store.zarr"
    (store / "arr").mkdir(parents=True)
    (store / "arr" / name).write_bytes(content)
    assert_refused(store, "unreadable Zarr metadata")


@pytest.mark.parametrize("name", [".zarray", "zarr.json"])
@pytest.mark.parametrize("content", ["[1, 2]", '"text"', "7", "null"])
def test_metadata_that_is_not_a_json_object_is_refused(tmp_path, name, content):
    store = tmp_path / "store.zarr"
    (store / "arr").mkdir(parents=True)
    (store / "arr" / name).write_text(content)
    assert_refused(store, "not a JSON object")


def test_a_missing_or_non_folder_store_is_refused(tmp_path):
    (tmp_path / "file").write_text("x")
    for path in (tmp_path / "missing", tmp_path / "file"):
        assert_refused(path, "is not a Zarr store folder")


@pytest.mark.parametrize(
    ("data_type", "codecs", "message"),
    [
        ("object", [{"name": "bytes"}], "data type 'object' is not allowed"),
        ({"name": "some_custom_type"}, [{"name": "bytes"}], "data type .* is not allowed"),
        (None, [{"name": "bytes"}], "data type None is not allowed"),
        ("float32", [{"name": "bytes"}, {"name": "numcodecs.pickle"}], "codec 'numcodecs.pickle' is not allowed"),
        ("float32", {"name": "bytes"}, "is not a list"),
        ("float32", [{"name": ["bytes"]}], "codec .* is not allowed"),
        ("float32", ["sharding_indexed"], "without a configuration object"),
        (
            "float32",
            [
                {
                    "name": "sharding_indexed",
                    "configuration": {"codecs": [{"name": "bytes"}], "index_codecs": [{"name": "x"}]},
                }
            ],
            "codec 'x' is not allowed",
        ),
        (
            "float32",
            [
                {
                    "name": "sharding_indexed",
                    "configuration": {"codecs": [{"name": "x"}], "index_codecs": [{"name": "bytes"}]},
                }
            ],
            "codec 'x' is not allowed",
        ),
    ],
)
def test_v3_refusals_say_why(tmp_path, data_type, codecs, message):
    assert_refused(v3_store(tmp_path, data_type, codecs), message)


@pytest.mark.parametrize(
    "configuration",
    [{"index_codecs": [{"name": "bytes"}]}, {"codecs": [{"name": "bytes"}]}, {}],
    ids=["no codecs", "no index_codecs", "neither"],
)
def test_a_sharding_codec_missing_its_inner_codec_lists_is_refused(tmp_path, configuration):
    # the Zarr v3 sharding spec requires both lists; a check that cannot see them must refuse, not assume []
    assert_refused(
        v3_store(tmp_path, codecs=[{"name": "sharding_indexed", "configuration": configuration}]),
        "sharding codec without",
    )


def test_a_sharding_codec_inside_a_sharding_codec_is_checked_too(tmp_path):
    inner = {
        "name": "sharding_indexed",
        "configuration": {"codecs": [{"name": "numcodecs.pickle"}], "index_codecs": []},
    }
    outer = {"name": "sharding_indexed", "configuration": {"codecs": [inner], "index_codecs": []}}
    assert_refused(v3_store(tmp_path, codecs=[outer]), "codec 'numcodecs.pickle' is not allowed")
