"""The `ftir-map` command: `info` summarizes a map, `export` writes it as a folder of files, `unpack` writes maps into
an open project folder and `catalog` lists that folder."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .omnic import OmnicMap, read_omnic
from .project import UNREADABLE, Project, open_project, unpack
from .unpacked import SPECTRA_FORMATS
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


def format_unpack_line(row: dict, action: str) -> str:
    """One line per source file: what happened, the map name, and what is in it (or why it was skipped)."""
    if action == "skipped":
        return f"skipped   {row['name']}: {row['status'].removeprefix('skipped: ')}"
    return f"{action:<9} {row['name']}: {row['n_points']} points, {row['map_kind']}, {row['n_images']} images"


def format_catalog(project: Project) -> str:
    """The catalog as a plain-text table: name, kind, points, axis range, data kind, images and status."""
    table = [("name", "map_kind", "n_points", "axis", "data_kind", "n_images", "status")]
    for row in project.rows:
        axis = "" if row["axis_min"] is None else f"{row['axis_min']:g}-{row['axis_max']:g} {row['axis_unit']}"
        cells = (row["name"], row["map_kind"], row["n_points"], axis, row["data_kind"], row["n_images"], row["status"])
        table.append(tuple("" if cell is None else str(cell) for cell in cells))
    widths = [max(len(line[i]) for line in table) for i in range(len(table[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(line, widths, strict=True)).rstrip() for line in table)


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
    pack = commands.add_parser("unpack", help="unpack .map files into an open project folder")
    pack.add_argument("sources", nargs="+", metavar="SRC", help=".map files and folders searched for .map files")
    pack.add_argument("--out", required=True, metavar="PROJECT", help="project folder")
    pack.add_argument(
        "--spectra", choices=SPECTRA_FORMATS, default="zarr", help="spectra file; nc needs the [netcdf] extra"
    )
    pack.add_argument("--png", action="store_true", help="also write the images as PNG")
    pack.add_argument("--copy-source", action="store_true", help="copy each original .map file into its folder")
    pack.add_argument("--overwrite", action="store_true", help="unpack again maps that are already in the project")
    catalog = commands.add_parser("catalog", help="list the maps of a project folder")
    catalog.add_argument("project")
    return parser


def run_unpack(args: argparse.Namespace) -> None:
    counts = dict.fromkeys(("unpacked", "unchanged", "skipped"), 0)

    def progress(row: dict, action: str) -> None:
        counts[action] += 1
        print(format_unpack_line(row, action))

    unpack(
        args.sources,
        args.out,
        spectra=args.spectra,
        png=args.png,
        copy_source=args.copy_source,
        overwrite=args.overwrite,
        progress=progress,
    )
    print(f"{counts['unpacked']} unpacked, {counts['unchanged']} unchanged, {counts['skipped']} skipped -> {args.out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "unpack":
            run_unpack(args)
        elif args.command == "catalog":
            print(format_catalog(open_project(args.project)))
        elif args.command == "info":
            print(format_info(read_omnic(args.file)))
        else:
            m = read_omnic(args.file)
            folder = export_all(m, args.out, args.spectra, args.metadata, args.points, args.images, args.overwrite)
            print(f"wrote {folder}")
    except (*UNREADABLE, ImportError) as error:
        message = str(error).replace("overwrite=True", "--overwrite")  # the CLI spelling of the Python argument
        print(f"ftir-map: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
