"""The `ftir-map` command: `info FILE` summarizes a map, `export FILE --out DIR` writes it as a folder of files."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .omnic import OmnicFormatError, OmnicMap, read_omnic
from .writers import IMAGE_FORMATS, METADATA_WRITERS, POINTS_WRITERS, SPECTRA_WRITERS, export_all

KEY_PARAMETERS = ("n_sample_scans", "n_background_scans", "data_spacing", "laser_frequency", "mirror_velocity")


def format_info(m: OmnicMap) -> str:
    """A short text summary: map kind, size, axis range, data kind, images and key collection parameters."""
    shape = f" ({m.grid_shape[0]} rows x {m.grid_shape[1]} columns)" if m.grid_shape else ""
    lines = [
        f"file: {m.path.name}",
        f"map kind: {m.map_kind}{shape}",
        f"points: {m.n_points}",
        f"axis: {m.axis.min():g} to {m.axis.max():g} {m.axis_unit}, {len(m.axis)} values",
        f"data kind: {m.data_kind}",
        f"images: {len(m.images)}",
        "collection:",
    ]
    lines += [f"  {name}: {getattr(m.collection, name)}" for name in KEY_PARAMETERS]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftir-map", description="Inspect and export Thermo OMNIC .map files.")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="print a summary of a map")
    info.add_argument("file")
    export = commands.add_parser("export", help="write spectra, points, metadata and images to a folder")
    export.add_argument("file")
    export.add_argument("--out", required=True, help="output folder")
    export.add_argument("--spectra", choices=sorted(SPECTRA_WRITERS), default="zarr")
    export.add_argument("--metadata", choices=sorted(METADATA_WRITERS), default="json")
    export.add_argument("--points", choices=sorted(POINTS_WRITERS), default="csv")
    export.add_argument("--images", choices=[*IMAGE_FORMATS, "none"], default="png")
    export.add_argument("--overwrite", action="store_true", help="write into a folder that is not empty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        m = read_omnic(args.file)
        if args.command == "info":
            print(format_info(m))
        else:
            folder = export_all(m, args.out, args.spectra, args.metadata, args.points, args.images, args.overwrite)
            print(f"wrote {folder}")
    except (OSError, ImportError, OmnicFormatError, ValueError) as error:
        print(f"ftir-map: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
