"""Command line entry point for the physical photoacoustic FEM front-end."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

from . import __version__
from .config import CaseConfig, load_config
from .geometry import generate_mesh
from .mesh_check import audit_mesh
from .run_case import solve_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2-D Gmsh + meshio + scikit-fem photoacoustic forward model")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    solve = sub.add_parser("solve", aliases=["run"], help="generate/audit mesh and run transient FEM")
    solve.add_argument("--config", type=Path, default=None,
                       help="JSON case file (default: built-in debug configuration)")
    solve.add_argument("--outdir", type=Path, required=True)
    solve.add_argument("--no-plots", action="store_true")
    solve.add_argument("--no-linearity-check", action="store_true",
                       help="skip the second scaled initial-value solve")
    solve.add_argument("--mesh", type=Path, default=None, help="reuse a pre-generated Gmsh .msh")
    mesh = sub.add_parser("mesh", help="generate and audit a Gmsh mesh only")
    mesh.add_argument("--config", type=Path, default=None,
                      help="JSON case file (default: built-in debug configuration)")
    mesh.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("audit", help="audit an existing .msh with meshio")
    audit.add_argument("mesh", type=Path)
    audit.add_argument("--config", type=Path, default=None)
    test = sub.add_parser("self-test", help="small real-Gmsh/scikit-fem smoke test")
    test.add_argument("--outdir", type=Path, default=None)
    test.add_argument("--keep", action="store_true", help="keep temporary self-test output")
    strict = sub.add_parser("strict-validate", help="run independent Stage A/B and 10 MHz benchmarks")
    strict.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in ("solve", "run"):
        cfg = load_config(args.config) if args.config else CaseConfig()
        summary = solve_case(cfg, args.outdir, make_plots=not args.no_plots,
                             validate_linearity=not args.no_linearity_check,
                             mesh_path=args.mesh, command=[sys.executable, *sys.argv[1:]])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["validation_status"] == "pass" else 2
    if args.command == "mesh":
        cfg = load_config(args.config) if args.config else CaseConfig()
        path = generate_mesh(cfg, args.out)
        data = audit_mesh(path, cfg)
        print(json.dumps(data.to_dict(), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "audit":
        cfg = load_config(args.config) if args.config else None
        data = audit_mesh(args.mesh, cfg)
        print(json.dumps(data.to_dict(), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "self-test":
        base = CaseConfig()
        quick = replace(base,
                        case_name="self_test",
                        mesh=replace(base.mesh, element_size_m=0.50e-3,
                                     interface_size_m=0.25e-3, source_size_m=0.20e-3),
                        time=replace(base.time, dt_s=50e-9, t_end_s=5.0e-6, save_every=5))
        if args.outdir is None and not args.keep:
            with tempfile.TemporaryDirectory(prefix="pa_fem_self_test_") as tmp:
                summary = solve_case(quick, Path(tmp), make_plots=False,
                                     validate_linearity=True,
                                     command=[sys.executable, *sys.argv[1:]])
                print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            target = args.outdir or Path("results/self_test")
            summary = solve_case(quick, target, make_plots=False,
                                 validate_linearity=True,
                                 command=[sys.executable, *sys.argv[1:]])
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["validation_status"] == "pass" else 2
    if args.command == "strict-validate":
        from .strict_validation import run_strict_validation

        report = run_strict_validation(args.outdir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "pass" else 2
    raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
