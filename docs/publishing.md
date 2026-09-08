# Publishing to PyPI

The release workflow is `.github/workflows/release.yml`. Building or testing
does not upload anything. Publishing requires an explicit workflow dispatch
with `publish=true` on a matching version tag.

## One-time account configuration

Log into PyPI, verify the account email and enable two-factor authentication.
Open [Publishing](https://pypi.org/manage/account/publishing/) and add a pending
GitHub publisher with these exact values:

| Field | Value |
|---|---|
| PyPI project name | `ultrafast-pycocotools` |
| Owner | `developer0hye` |
| Repository name | `ultrafast-pycocotools` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The workflow field is the filename, not the workflow display name or full path.
Pending publishers create the project on the first successful upload; they do
not reserve its name. See the [official PyPI instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
No long-lived API token or password is stored in GitHub. Only the upload job has
OIDC permission, using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

The GitHub environment `pypi` allows version tags matching `v*`. The workflow
also checks the exact tag against the Cargo workspace version, main ancestry
and a successful `CI` check on the exact commit before allowing publication.

## Build and verify

The workflow builds **34 CPython wheels**: Python 3.8–3.14 for Linux x86-64/ARM64,
Windows x86-64, macOS Intel and macOS Apple Silicon. Linux wheels use
manylinux2014 (glibc 2.17+). Apple Silicon wheels start at Python 3.9. Free-threaded CPython and musl wheels
are not included in this release matrix.

Every wheel is installed directly from its artifact without rebuilding the
package and runs the existing pycocotools comparison suite. The separate
Library CI covers optional LVIS and RF-DETR dependencies. A source archive is
also rebuilt and tested independently. The final verification checks all 34
wheel tags, package versions, required Python/native files, source archive
contents and strict PyPI metadata validity, then records SHA-256 checksums.

For a build without upload:

```bash
gh workflow run release.yml --ref main -f publish=false
```

Branches named `release/**` also build automatically without uploading. Download
the tested wheels from the run's `distribution-*` artifacts and install with:

```bash
python -m pip install --only-binary=ultrafast-pycocotools \
  --find-links=dist ultrafast-pycocotools==0.1.6
```

Installing a compatible wheel requires no Rust compiler. Installing the source
archive requires a Rust/compiler toolchain. Matching wheels are selected by pip.

## Publish a reviewed version

After the build and Library CI pass, and the PyPI publisher is configured:

1. Ensure the Cargo workspace version and lockfile match the intended release.
2. Create and push `v<version>` at the tested commit on main.
3. Explicitly run the release workflow on that tag with `publish=true`.

For example, for version 0.1.6:

```bash
git tag v0.1.6
git push origin v0.1.6
gh workflow run release.yml --ref v0.1.6 -f publish=true
```

The upload job publishes the exact artifacts that passed the build/test jobs in
that run. It does not rebuild them or silently skip existing files. PyPI does
not allow overwriting a released distribution: fixes require a new version.
After publication, verify installation from PyPI in a fresh environment:

```bash
python -m pip install --only-binary=ultrafast-pycocotools ultrafast-pycocotools==0.1.6
python -c "import importlib.metadata as m; import ultrafast_pycocotools._ufcoco; print(m.version('ultrafast-pycocotools'))"
```

## Release history

Version 0.1.4 published all 24 wheels, but PyPI rejected its source archive
because the declared LICENSE file was missing from the archive. Version 0.1.5
adds that file explicitly and checks declared license paths before upload.
The evaluator implementation is unchanged from 0.1.4.

Version 0.1.6 adds Python 3.8 and Linux ARM64 wheels, plus an explicit
`lvis_protocol="coco"` mode that retains the federated per-category detection
limits and COCO crowd semantics used by faster-coco-eval. The default official
LVIS protocol and default pycocotools metrics remain unchanged.
