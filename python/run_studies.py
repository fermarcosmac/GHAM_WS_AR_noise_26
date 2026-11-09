"""Run configured FSM/Python and PL/MATLAB studies from the repository root.

The default runs both benchmarks and both parametrizations. PL calls the existing
MATLAB runners; BI_HMC is computed once in Python and reused by MATLAB analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("example_1", "example_CSTR")
STUDIES = ("montecarlo", "initialization", "noise", "record-length")


def matlab_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_commands(args):
    """Return argument arrays; no shell parsing or experiment execution here."""
    commands = []
    benchmarks = BENCHMARKS if args.experiment == "both" else (args.experiment,)
    study_config = args.study_config.resolve()
    for benchmark in benchmarks:
        if args.parametrization in ("fsm", "both"):
            script = "comparison_torch.py" if args.study == "montecarlo" else args.study.replace("-", "_") + "_study.py"
            command = [sys.executable, str(ROOT / "python" / script), "--experiment", benchmark]
            if args.study == "montecarlo":
                command += ["--mode", "MONTECARLO", "--no-show"]
            else:
                command += ["--study-config", str(study_config)]
            if args.cpu:
                command += ["--cpu"]
            for flag in ("runs", "iterations", "hmc_config"):
                value = getattr(args, flag)
                if value is not None:
                    command += ["--" + flag.replace("_", "-"), str(value)]
            commands.append(command)
        if args.parametrization not in ("pl", "both"):
            continue
        mode = {"montecarlo": "MONTECARLO", "initialization": "INITIALIZATION"}.get(args.study)
        library = None
        if mode:
            commands.append([sys.executable, str(ROOT / "python" / "export_shared_python_records.py"),
                             "--experiment", benchmark, "--mode", mode])
            library = ROOT / "results" / benchmark / ("shared_python_" + mode.lower() + ".mat")
        setup = "addpath(" + matlab_string((ROOT / "mlab").as_posix()) + "); "
        if args.study == "montecarlo":
            setup += "cfg=jsondecode(fileread(" + matlab_string((ROOT / "configs" / (benchmark + ".json")).as_posix()) + ")); "
            setup += "cfg.mode='MONTECARLO'; cfg.make_local_plots=false; "
            if args.runs is not None:
                setup += f"cfg.N_MC={args.runs}; "
            if args.iterations is not None:
                setup += f"cfg.settings.K_max={args.iterations}; "
            setup += "comparison(cfg,struct('experiment_name'," + matlab_string(benchmark)
            setup += ",'make_local_plots',false,'python_record_library'," + matlab_string(library.as_posix()) + "));"
        else:
            overrides = {}
            if args.runs is not None:
                overrides["n_runs"] = args.runs
            if args.iterations is not None:
                overrides["K_max"] = args.iterations
            setup += "options=struct('study_config'," + matlab_string(study_config.as_posix())
            setup += ",'study',jsondecode(" + matlab_string(json.dumps(overrides)) + ")); "
            if library is not None:
                setup += "options.python_record_library=" + matlab_string(library.as_posix()) + "; "
            setup += args.study.replace("-", "_") + "_study(" + matlab_string(benchmark) + ",options);"
        commands.append([args.matlab, "-batch", setup])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES, required=True)
    parser.add_argument("--experiment", choices=(*BENCHMARKS, "both"), default="both")
    parser.add_argument("--parametrization", choices=("fsm", "pl", "both"), default="both")
    parser.add_argument("--study-config", type=Path, default=ROOT / "configs" / "studies.json")
    parser.add_argument("--runs", type=int, help="Override the repetition count in both backends")
    parser.add_argument("--iterations", type=int, help="Override optimizer iteration limits in both backends")
    parser.add_argument("--hmc-config", type=Path, help="Flat JSON sampler overrides, applied in Python")
    parser.add_argument("--cpu", action="store_true", help="Run the Python optimization methods on CPU")
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable name or full path")
    parser.add_argument("--dry-run", action="store_true", help="Print all calls without running or writing results")
    args = parser.parse_args(argv)
    for key in ("runs", "iterations"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key} must be positive")
    if args.hmc_config is not None:
        args.hmc_config = args.hmc_config.resolve()
    commands = build_commands(args)
    if not args.dry_run and args.parametrization in ("pl", "both") and not shutil.which(args.matlab):
        parser.error("MATLAB was not found; add it to PATH or pass --matlab with its full executable path")
    for index, command in enumerate(commands, 1):
        display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        print(f"[{index}/{len(commands)}] " + display, flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    return commands


if __name__ == "__main__":
    main()
