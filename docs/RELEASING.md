# Releasing

Releases are built and uploaded by `.github/workflows/publish.yml` when a tag starting with `v` is pushed. It builds
with `uv build`, runs `uvx twine check dist/*`, and uploads with PyPI trusted publishing, so no PyPI token is stored
anywhere.

## One-time setup on pypi.org

The project already exists on PyPI, so add the publisher to it: pypi.org > Your projects > ftir-omnic-map-reader >
Manage > Publishing > Add a new publisher > GitHub. Enter:

| Field | Value |
|---|---|
| PyPI project name | `ftir-omnic-map-reader` (the name PyPI shows; `ftir-OMNIC-map-reader` is the same name) |
| Owner | `whtowbin` |
| Repository name | `FTIR_OMNIC_MAP_Reader` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

On GitHub, create the environment: repository > Settings > Environments > New environment > `pypi`. Optionally add
yourself as a required reviewer, so each upload needs a click.

## Steps for each release

1. Merge the work into `main`. Check that the `tests` workflow is green on it.
2. Set `version` in `pyproject.toml` and add the entry to `CHANGELOG.md`. For 1.1.0 both are already done.
3. Build locally and look at what goes in:

   ```console
   uv build
   uvx twine check dist/*
   unzip -l dist/*.whl          # package code only, no .map files
   tar tzf dist/*.tar.gz        # no tests/data
   ```

4. Commit, then tag and push the tag. The tag must match the version:

   ```console
   git tag v1.1.0
   git push origin main v1.1.0
   ```

5. Watch the `publish` workflow in the Actions tab. After approval (if you set a reviewer) it uploads to PyPI.
6. Check the new version on pypi.org and install it in a clean environment:
   `uv run --with ftir-OMNIC-map-reader==1.1.0 --no-project python -c "import ftir_map; print(ftir_map.read_omnic)"`.
7. Optionally create a GitHub release from the tag.

PyPI never accepts the same version twice. If a release is wrong, fix it and publish the next version (1.1.1).

## Notes

- Python 3.10 reached end of life in October 2026. The floor stays at `>=3.10` for 1.1; raise it in a later release.
- A trial run against TestPyPI needs a second trusted publisher on test.pypi.org and a copy of the workflow that
  sets `repository-url: https://test.pypi.org/legacy/`; this repository does not include one.
