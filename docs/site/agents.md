# Using with AI coding agents

The repository includes a skill for AI coding agents:
[`.claude/skills/ftir-map/SKILL.md`](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/blob/main/.claude/skills/ftir-map/SKILL.md).
It gives an agent what it needs to use this package without reading the source:

- how to read a map, and the fields of `OmnicMap` with their types and shapes;
- how the [map kind](map_kinds.md) decides the shapes and dimension names (grid, line, points, array, interferogram);
- recipes: a band-area image of a grid map, a profile along a line scan, positions over the video image or mosaic;
- the exports, unpacked projects, the command line and the version 1.0 `Load_Omnic_Map`;
- the traps: `m.spectra` is always in file order, row 0 of a grid is the smallest stage Y, NaN is kept, interferogram
  maps have no wavenumber axis.

Every example in the skill was run against the maps in `tests/data`, and an agent given only the skill completed
typical tasks on them (band-area images, line profiles, image overlays, unpacking a folder of maps).

The skill covers the package API, not spectroscopy: it does not choose baselines or interpret bands.

## Claude Code

In a clone of this repository, Claude Code finds the skill by itself. To use it in your own project, copy the folder
into that project's `.claude/skills/`, or into `~/.claude/skills/` to have it in every project:

```bash
mkdir -p ~/.claude/skills/ftir-map
curl -L -o ~/.claude/skills/ftir-map/SKILL.md \
  https://raw.githubusercontent.com/whtowbin/FTIR_OMNIC_MAP_Reader/main/.claude/skills/ftir-map/SKILL.md
```

Claude Code loads the skill when a task involves OMNIC `.map` files or `ftir_map`.

## Other agents

`SKILL.md` is plain Markdown with a short YAML header (`name`, `description`). Agents that read skills in this format
can use the folder as it is. For any other agent, put the file in the project and point to it from the agent's
instructions (for example `AGENTS.md`), or paste it into the conversation.

## Keeping it current

The skill describes the API of the version on `main`. Check `ftir_map.__version__` against the changelog if an
example does not work, and report a mismatch in the
[issue tracker](https://github.com/whtowbin/FTIR_OMNIC_MAP_Reader/issues).
