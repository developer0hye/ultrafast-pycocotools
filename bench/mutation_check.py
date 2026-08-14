"""Re-run the mutations an audit found surviving, and check they now fail.

A test suite is only worth its mutation score. This applies each mutation to
the Rust core, runs `cargo test -p ufcoco-core`, records which tests caught it,
and reverts — so "we fixed the coverage hole" is a measurement rather than a
claim.

Rust-only on purpose: `ufcoco-core` builds and tests without Python, so a full
sweep costs seconds instead of a rebuild per mutation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "rust" / "ufcoco-core" / "src"

# (label, file, exact text to find, replacement)
MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "drop the np.spacing epsilon",
        "eval.rs",
        "buf.pr[n] = tpf / (fpf + tpf + EPS);",
        "buf.pr[n] = tpf / (fpf + tpf);",
    ),
    (
        "delete the precision envelope",
        "eval.rs",
        """            for i in (1..nd).rev() {
                if buf.pr[i] > buf.pr[i - 1] {
                    buf.pr[i - 1] = buf.pr[i];
                }
            }""",
        "            // envelope removed",
    ),
    (
        "reverse the envelope direction",
        "eval.rs",
        "for i in (1..nd).rev() {",
        "for i in 1..nd {",
    ),
    (
        "searchsorted side=left -> right",
        "eval.rs",
        "while pi < nd && buf.rc[pi] < thr {",
        "while pi < nd && buf.rc[pi] <= thr {",
    ),
    (
        "recall reads the first point",
        "eval.rs",
        "if nd > 0 { buf.rc[nd - 1] } else { 0.0 };",
        "if nd > 0 { buf.rc[0] } else { 0.0 };",
    ),
    (
        "accumulate ignores the maxDets cut",
        "eval.rs",
        "let d_n = mm.dt_scores.len().min(max_det);",
        "let d_n = mm.dt_scores.len();",
    ),
    (
        "prepare_category truncates to max_dets.first()",
        "eval.rs",
        "let max_det = self.params.max_dets.last().copied().unwrap_or(0);",
        "let max_det = self.params.max_dets.first().copied().unwrap_or(0);",
    ),
    (
        "run-out-of-recall writes -1 instead of 0",
        "eval.rs",
        """                } else {
                    out.precision[dst] = 0.0;
                    out.scores[dst] = 0.0;
                }""",
        """                } else {
                    out.precision[dst] = -1.0;
                    out.scores[dst] = -1.0;
                }""",
    ),
    (
        "remove the crowd re-match exemption",
        "eval.rs",
        """                        if gt_matched[tind * g_n + gind]
                            && !self.gt.iscrowd[ci.gt_idx[gsrc] as usize]
                        {
                            continue;
                        }""",
        """                        if gt_matched[tind * g_n + gind] {
                            continue;
                        }""",
    ),
    (
        "remove the ignore-sorted early break",
        "eval.rs",
        """                        if m > -1 && !gt_ignore[m as usize] && gt_ignore[gind] {
                            break;
                        }""",
        "                        // break removed",
    ),
    (
        "IoU match uses <= instead of <",
        "eval.rs",
        """                        if v < best {
                            continue;
                        }""",
        """                        if v <= best {
                            continue;
                        }""",
    ),
    (
        "drop the 1 - 1e-10 threshold clamp",
        "eval.rs",
        "let mut best = f64::min(thr, 1.0 - 1e-10);",
        "let mut best = thr;",
    ),
    (
        "OKS drops the per-keypoint mean",
        "eval.rs",
        "out[i * n + j] = sum / cnt as f64;",
        "out[i * n + j] = sum;",
    ),
    (
        "OKS drops the sigma doubling",
        "eval.rs",
        ".map(|s| (s * 2.0) * (s * 2.0))",
        ".map(|s| s * s)",
    ),
    (
        "c_i32 returns 0 instead of INT_MIN",
        "rle.rs",
        """    } else {
        i32::MIN
    }
}""",
        """    } else {
        0
    }
}""",
    ),
    (
        "area sums the even runs",
        "rle.rs",
        "let mut j = 1;\n        while j < self.cnts.len() {",
        "let mut j = 0;\n        while j < self.cnts.len() {",
    ),
    (
        "to_string delta starts one index early",
        "rle.rs",
        "if i > 2 {\n                x -= self.cnts[i - 2] as i64;",
        "if i > 1 {\n                x -= self.cnts[i - 2] as i64;",
    ),
    (
        "from_str drops sign extension",
        "rle.rs",
        """                    let shift = 5 * k;
                    if shift < 64 {
                        x |= -1i64 << shift;
                    }""",
        "                    // sign extension removed",
    ),
    (
        "bb_iou output transposed",
        "rle.rs",
        "out[d * n + g] = o;",
        "out[g * m + d] = o;",
    ),
    (
        "to_bbox drops the even truncation",
        "rle.rs",
        "let m = (self.cnts.len() / 2) * 2;",
        "let m = self.cnts.len();",
    ),
    (
        "boundary dilation floor removed",
        "rle.rs",
        "if dilation < 1 {\n        dilation = 1;\n    }",
        "if dilation < 0 {\n        dilation = 0;\n    }",
    ),
]


def run_tests() -> tuple[bool, list[str]]:
    proc = subprocess.run(
        ["cargo", "test", "-p", "ufcoco-core", "--lib"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    failed = [
        line.split()[1]
        for line in proc.stdout.splitlines()
        if line.startswith("test ") and line.rstrip().endswith("FAILED")
    ]
    return proc.returncode == 0, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter on the mutation label")
    args = ap.parse_args()

    ok, _ = run_tests()
    if not ok:
        sys.exit("baseline is red; fix the suite before mutating it")

    survived = []
    print(f"{'mutation':46} {'caught by':>10}  first failing test")
    print("-" * 100)
    for label, fname, old, new in MUTATIONS:
        if args.only and args.only not in label:
            continue
        path = CORE / fname
        original = path.read_text(encoding="utf-8")
        if old not in original:
            print(f"{label:46} {'SKIP':>10}  pattern not found in {fname}")
            continue
        try:
            path.write_text(original.replace(old, new, 1), encoding="utf-8")
            passed, failing = run_tests()
        finally:
            path.write_text(original, encoding="utf-8")
        if passed:
            survived.append(label)
            print(f"{label:46} {'SURVIVED':>10}  <-- coverage hole")
        else:
            print(f"{label:46} {len(failing):>10}  {failing[0] if failing else '(build error)'}")

    total = len([m for m in MUTATIONS if not args.only or args.only in m[0]])
    print()
    print(f"caught {total - len(survived)}/{total}")
    for label in survived:
        print(f"  SURVIVED: {label}")


if __name__ == "__main__":
    main()
