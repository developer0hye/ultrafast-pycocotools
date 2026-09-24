# Automated regression checks

[Workflow runs](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml)
are defined in [ci.yml](../.github/workflows/ci.yml).

Every push and pull request runs the full test suite. Weekly and manual runs
also check that a fresh source build still works with current dependencies.
The upstream integration jobs live in
[upstream-integrations.yml](../.github/workflows/upstream-integrations.yml):
`ci.yml` calls them with pinned RF-DETR and Ultralytics commits as part of the
required `CI` status, and
[upstream-canary.yml](../.github/workflows/upstream-canary.yml) calls them
weekly with the latest upstream revisions. Both install the upstream project
first and then replace its released ultrafast-pycocotools wheel with a build of
this revision; `scripts/check_integration_build.py` fails the job if that
replacement did not happen or the upstream revision no longer uses ultrafast.
Actions are pinned to commit SHAs, and jobs have read-only repository access.

| Check | Coverage |
|---|---|
| Linux | Python 3.9 / NumPy 1; Python 3.12 / NumPy 1 and 2; Python 3.14 / NumPy 2 |
| macOS and Windows | Python 3.12 / NumPy 2 |
| Rust core | Unit tests in debug and optimized release builds |
| LVIS on every OS | Official 100-image example, bbox/segmentation full precision and recall arrays plus all 13 summary metrics; federated synthetic edge cases |
| Scorer parity | Complete precision, recall, scores and stats arrays; bbox, segmentation and keypoints |
| Edge cases and API | Crowds, tied scores, area boundaries, RLE/masks, query ordering, subclass overrides and diagnostics |
| Determinism | Comparison across Rayon thread counts |
| Reproduction | Fresh synthetic inputs, both in-memory and compact file loading, isolated backend processes, published hashes on the recorded reference environment |
| RF-DETR integration | CPU job on RF-DETR's `ufcoco` backend commit (roboflow/rf-detr#1449): this repository's adapter tests (bbox/segmentation metrics, lifecycle, two-rank Gloo merge) and RF-DETR's own COCO metric, evaluation-callback and trainer tests, with this revision replacing the released wheel |
| Ultralytics integration | CPU job on the head of ultralytics/ultralytics#26101: YOLO26n detect/segment/pose validation on coco8 through Ultralytics' in-memory and file evaluation routes, with complete arrays, statistics and reported metrics compared to pycocotools, plus Ultralytics' own evaluator test. Datasets and weights are pinned by SHA-256 and Ultralytics runs offline |
| Upstream canary | Weekly, non-blocking: the same two jobs against RF-DETR `develop` and the latest #26101 head |
| Additional backend | Linux / Python 3.12 checks faster-coco-eval 1.8.0 on identical generated inputs; numerical agreement is distinct from byte identity |
| Benchmark integrity | Changed inputs fail before scoring; one-ULP array changes fail; nested scaling subsets preserve input order and categories |

The source distribution is compiled on every platform; importing the native
extension and reference package is mandatory before tests start. The bundled
real COCO fixture is always tested. Three optional tests needing full local
COCO files skip on hosted runners; the synthetic and bundled-fixture checks do
not need GPUs. Every Python matrix job additionally downloads the official LVIS
example JSON files (~5.5 MiB total), checks pinned SHA-256 hashes, and runs the
LVIS tests against lvis 0.5.3. A failed download/hash check fails the job.
No image files are downloaded. `lvis-test` dependencies are installed on all
three operating systems, with Matplotlib using its noninteractive Agg backend. Logs and JUnit reports are retained for
14 days, including on failure. Timing is not a pass/fail threshold on shared
hosted runners.

The required **CI** check succeeds only when every Python matrix entry, both
Rust builds and the RF-DETR integration job succeed. Failed, cancelled or skipped prerequisite jobs make that
check fail. The `main` branch requires this check on an up-to-date commit;
administrators are included in enforcement. Force pushes and branch deletion
are disabled. Changes are tested on a branch before main is updated; pull requests use the
same required check.

This follows GitHub's [Python testing guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and [required status-check protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
Tests guard the covered behavior; they cannot prove the absence of every bug.
New fixes should add a regression case that fails before the fix.
