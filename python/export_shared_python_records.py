"""Recover the saved Python study records for MATLAB PL, without fitting methods."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat, savemat

from diagnostics import array_digest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("example_1", "example_CSTR")
SOURCE_FILES = {
    "MONTECARLO": "comparison_fsm_montecarlo.mat",
    "INITIALIZATION": "initialization_fsm.mat",
}


def field(struct, name):
    # scipy may leave structs inside the all_runs cell array as mat_structs.
    return struct[name] if isinstance(struct, dict) else getattr(struct, name)


def replay_montecarlo(experiment, metadata, reference, context):
    import torch

    from wiener_system import WienerDimensions, generate_example_data, simulate_wiener

    device = torch.device(field(metadata, "device"))
    dtype = torch.float64
    burn_in = int(experiment["burn_in"])
    torch.manual_seed(int(field(metadata, "data_seed")))
    dims = WienerDimensions(**experiment["dims"])
    theta = torch.as_tensor(experiment["theta_true"], dtype=dtype, device=device)
    with torch.no_grad():
        data = generate_example_data(
            dims, theta, int(experiment["lambda_g"]), burn_in,
            float(experiment["sigma_nu"]), device=device, dtype=dtype,
        )
        # Check the innovations against an archived BI_HMC output trace by
        # evaluating the saved estimate, without running the sampler.
        saved_output = np.asarray(field(reference, "output_estimate"))
        if saved_output.size and np.isfinite(saved_output).all():
            estimate = torch.as_tensor(field(reference, "theta_hat"), dtype=dtype, device=device)
            reconstructed = simulate_wiener(data["r"], data["nu"], estimate, dims)
            if array_digest(reconstructed) != array_digest(saved_output):
                raise ValueError(f"Innovations do not reproduce the saved BI_HMC output: {context}")
    record = {key: data[key].cpu().numpy() for key in ("r", "nu", "c")}
    record["input_prehistory"] = data["r_full"][:burn_in].cpu().numpy()
    return record


def export_records(source_file: Path, mode: str) -> Path:
    payload = loadmat(source_file, simplify_cells=True)
    experiment = payload["experiment"]
    if payload["parametrization"] != "FSM" or experiment["mode"] != mode:
        raise ValueError(f"Unexpected study in {source_file}")

    saved_record = payload["example_data"]
    keys = ("r", "nu", "c", "input_prehistory")
    records = []
    dataset_ids, data_seeds = [], []
    runs = np.atleast_1d(payload["all_runs"])
    if len(runs) != experiment["N_MC"]:
        raise ValueError(f"Incomplete run history in {source_file}")

    initialization = {}
    if mode == "INITIALIZATION":
        study = payload["study"]
        starts = np.asarray(study["initializations"]).reshape(-1, len(runs))
        initialization = {"initializations": starts, "initialization_seed": study["initialization_seed"]}

    for index, run in enumerate(runs, 1):
        context = f"{source_file}, run {index}"
        methods = np.atleast_1d(run)
        reference = next(method for method in methods if field(method, "name") == "BI_HMC")
        metadata = field(reference, "run_metadata")
        seed = int(field(metadata, "data_seed"))
        index_key = "mc_index" if mode == "MONTECARLO" else "initialization_index"
        expected_seed = experiment["seed"] + (index - 1 if mode == "MONTECARLO" else 0)
        if field(metadata, index_key) != index or seed != expected_seed:
            raise ValueError(f"Unexpected run index or seed: {context}")

        if mode == "MONTECARLO":
            # Use the data generator from comparison_torch, with the recorded
            # PyTorch device and seed. The first run also anchors the innovations.
            optimizer = next(method for method in methods if field(method, "name") != "BI_HMC")
            record = replay_montecarlo(experiment, field(optimizer, "run_metadata"), reference, context)
        else:
            record = {key: np.asarray(saved_record[key]).reshape(-1) for key in keys}
            if array_digest(starts[:, index-1]) != field(metadata, "initialization_sha256"):
                raise ValueError(f"Initialization does not match BI_HMC: {context}")

        if index == 1 and any(array_digest(record[key]) != array_digest(saved_record[key]) for key in keys):
            raise ValueError(f"First replayed record differs from the saved signals: {context}")
        dataset_id = array_digest(record["r"], record["c"])
        if dataset_id != field(metadata, "dataset_id") or dataset_id != field(metadata, "data_sha256"):
            raise ValueError(f"Input/output SHA-256 mismatch: {context}")
        if array_digest(record["input_prehistory"]) != field(metadata, "input_history_sha256"):
            raise ValueError(f"Input prehistory SHA-256 mismatch: {context}")
        if any(field(field(method, "run_metadata"), "dataset_id") != dataset_id for method in methods):
            raise ValueError(f"Saved Python methods used different records: {context}")
        dataset_ids.append(dataset_id)
        data_seeds.append(seed)
        if index == 1 or mode == "MONTECARLO":
            records.append(record)

    if len(set(dataset_ids)) != len(records):
        raise ValueError(f"Unexpected number of distinct records in {source_file}")
    library = {key: np.column_stack([record[key] for record in records]) for key in keys}
    library.update(
        schema_version=1, experiment=experiment, source_fsm_file=source_file.name,
        source_generated_at=payload["generated_at"], data_seeds=data_seeds,
        dataset_ids=np.asarray(dataset_ids, dtype=object),
        innovation_sha256=np.asarray([array_digest(record["nu"]) for record in records], dtype=object),
        input_history_sha256=np.asarray([array_digest(record["input_prehistory"]) for record in records], dtype=object),
        **initialization,
    )
    output = source_file.with_name(f"shared_python_{mode.lower()}.mat")
    with NamedTemporaryFile(dir=output.parent, suffix=".mat", delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        savemat(staging, library, do_compression=True, long_field_names=True)
        staging.replace(output)
    finally:
        staging.unlink(missing_ok=True)
    print(f"{output}: verified {len(runs)} Python runs, exported {len(records)} distinct records")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=BENCHMARKS, action="append", help="Default: both benchmarks")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--mode", choices=tuple(SOURCE_FILES), action="append",
                        help="Export only the selected study; default: Monte Carlo and initialization")
    args = parser.parse_args(argv)
    for benchmark in args.experiment or BENCHMARKS:
        for mode in args.mode or SOURCE_FILES:
            export_records(args.results_dir / benchmark / SOURCE_FILES[mode], mode)


if __name__ == "__main__":
    main()
