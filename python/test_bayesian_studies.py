"""Checks experimental boundaries and sampler validity in the study adapters."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import loadmat

from comparison_torch import build_dims_and_theta, load_experiment_config
from identification_studies import (_run, _setup, bias_statistics, main, prefix_data, reference_data,
                                    run_initialization_study, run_noise_study, run_record_length_study,
                                    terminal_arrays, terminal_metadata)
from study_progress import StudyProgress


class BayesianStudyTests(unittest.TestCase):
    def test_cli_sampler_override_merges_over_study_file_without_running_inference(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="bi_cli_test_", dir=workspace/"python") as folder:
            directory = Path(folder).resolve()
            self.assertTrue(directory.is_relative_to(workspace))
            study_path, hmc_path = directory/"studies.json", directory/"hmc.json"
            study_path.write_text(json.dumps({"noise_study": {"n_runs": 1, "sigma_nu_values": [.1],
                "bayesian_hmc": {"chains": 4, "warmup": 90}}}), encoding="utf-8")
            hmc_path.write_text(json.dumps({"warmup": 2, "samples": 8}), encoding="utf-8")
            arguments = ["noise_study.py", "--experiment", "example_1", "--methods", "BI_HMC",
                         "--study-config", str(study_path), "--hmc-config", str(hmc_path), "--cpu"]
            with patch.object(sys, "argv", arguments), patch("identification_studies.run_noise_study") as run:
                main("noise")
            study = run.call_args.args[1]
            self.assertEqual(study["methods"], ["BI_HMC"])
            self.assertEqual(study["bayesian_hmc"], {"chains": 4, "warmup": 2, "samples": 8})

    def test_study_sampler_overrides_are_local_and_method_selection_is_exact(self):
        cfg = copy.deepcopy(load_experiment_config("example_1"))
        original_cfg = copy.deepcopy(cfg)
        study = {"methods": ["BI_HMC"], "bayesian_hmc": {"warmup": 3, "samples": 8,
                                                                   "initial_optimization_steps": 0}}
        original_study = copy.deepcopy(study)
        resolved, _, _, methods, configs = _setup(cfg, study, torch.device("cpu"))
        self.assertEqual(methods, ["BI_HMC"])
        self.assertFalse(resolved["include_python_extra_methods"])
        self.assertEqual(configs["BI_HMC"]["warmup"], 3)
        self.assertEqual(configs["BI_HMC"]["samples"], 8)
        self.assertEqual(configs["BI_HMC"]["initial_optimization_steps"], 0)
        self.assertEqual(cfg, original_cfg)
        self.assertEqual(study, original_study)
        with self.assertRaisesRegex(ValueError, "flat object"):
            _setup(cfg, {"bayesian_hmc": {"hmc": {"samples": 3}}}, torch.device("cpu"))

    def test_real_hmc_runs_all_three_studies_and_exports_final_scores_and_diagnostics(self):
        """A tiny budget tests plumbing, not posterior convergence or accuracy."""
        cfg = copy.deepcopy(load_experiment_config("example_1"))
        cfg["settings"].update(lambda_g=9, burn_in=12, K_max=2)
        cfg["diagnostics"] = {"enabled": False, "save_matrices": False}
        hmc = {"chains": 2, "warmup": 4, "samples": 8, "leapfrog_steps": 2,
               "initial_optimization_steps": 0, "step_size": .001,
               "adapt_mass": False, "adapt_step_size": False,
               "initial_jitter": .01, "progress_every": 3, "seed": 37}
        workspace = Path(__file__).resolve().parents[1]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(prefix="bi_study_test_", dir=workspace/"python") as folder:
                directory = Path(folder).resolve()
                self.assertTrue(directory.is_relative_to(workspace))
                config_path = directory/"methods.json"
                config_path.write_text(json.dumps({"methods": {"BI_HMC": hmc}}), encoding="utf-8")
                cfg["python_optim_config"] = str(config_path)
                common = {"methods": ["BI_HMC"], "n_runs": 1, "data_seed": 101,
                          "progress": False, "save_case_results": True}
                with redirect_stdout(io.StringIO()):
                    init_path = run_initialization_study(cfg, {**common, "n_runs": 2,
                        "initialization_seed": 111, "scale": .03}, output_file=directory/"init.mat")
                    noise_path = run_noise_study(cfg, {**common, "sigma_nu_values": [.05, .2]},
                                                 output_file=directory/"noise.mat")
                    record_path = run_record_length_study(cfg, {**common, "record_lengths": [6, 9],
                        "fsm_resolutions": [{"label": "exact"}]}, output_file=directory/"record.mat")
                initialization = loadmat(init_path, simplify_cells=True)
                self.assertEqual(initialization["experiment"]["selected_methods"], "BI_HMC")
                self.assertTrue(np.isfinite(initialization["mc_summary"]["final_param_estimates"]).all())
                self.assertEqual(initialization["example_data"]["input_prehistory"].size, 12)
                initial_seeds = np.asarray(initialization["study"]["sampling_seeds"]).reshape(-1)
                self.assertNotEqual(initial_seeds[0], initial_seeds[1])
                for path in (noise_path, record_path):
                    saved = loadmat(path, simplify_cells=True)["study"]
                    self.assertTrue(np.isfinite(saved["final_param_estimates"]).all())
                    self.assertTrue(np.all(saved["total_wall_time"] > 0))
                    self.assertTrue(np.asarray(saved["is_bayesian"]).all())
                    self.assertTrue(np.isin(saved["statuses"], ["sampling_complete", "sampling_warning"]).all())
                    self.assertEqual(saved["sampling_seeds"].reshape(-1)[0],
                                     saved["sampling_seeds"].reshape(-1)[1])
                    for case_path in np.asarray(saved["case_files"]).reshape(-1):
                        case = loadmat(str(case_path), simplify_cells=True)
                        result = case["results"]
                        self.assertEqual(result["theta_hist"].size, 0)
                        self.assertEqual(result["posterior_samples"].shape, (2, 8, 6))
                        self.assertEqual(result["run_metadata"]["input_history_length"], 12)
                        self.assertEqual(result["run_metadata"]["noise_burn_in"], 12)
                        self.assertEqual(result["run_metadata"]["noise_initialization"], "finite_burn_in")
                        self.assertTrue(np.isfinite(result["final_err"]))
        finally:
            torch.set_num_threads(previous_threads)

    def test_record_prefix_preserves_known_prehistory_and_routes_it_to_inference(self):
        cfg = copy.deepcopy(load_experiment_config("example_1"))
        cfg["settings"].update(lambda_g=12, burn_in=7, K_max=2)
        dims, theta = build_dims_and_theta(cfg, dtype=torch.float64, device=torch.device("cpu"))
        data = reference_data(cfg, dims, theta, 12, 1234, torch.device("cpu"))
        expected = np.random.default_rng(1234).standard_normal(19)
        np.testing.assert_array_equal(data["input_prehistory"], expected[:7])
        np.testing.assert_array_equal(data["r"], expected[7:])
        short = prefix_data(data, 4)
        self.assertEqual(short["r"].numel(), 4)
        self.assertEqual(short["input_prehistory"].numel(), 7)
        self.assertEqual(short["noise_burn_in"], 7)
        quiet = StudyProgress("test", ["BI_HMC"], 1, [4], 1, {"progress": False})
        with patch("identification_studies.run_method_set", return_value=[]) as dispatch:
            _run(cfg, ["BI_HMC"], {}, dims, theta, short, {}, False, quiet)
        context = dispatch.call_args.kwargs["data_context"]
        self.assertIs(context["input_prehistory"], data["input_prehistory"])
        self.assertEqual(context["noise_burn_in"], 7)

    def test_sampler_warnings_remain_distinct_from_passing_diagnostics(self):
        def result(status, estimate, valid=True):
            return {"name": "BI_HMC", "method_kind": "bayesian", "status": status,
                    "theta_hat": np.array([estimate]), "point_estimate_valid": valid,
                    "resolved_config_json": "{}", "run_metadata": {"dataset_id": "fixed", "sampler_seed": 45},
                    "total_time": 2., "total_wall_time": 3.,
                    "diagnostics_passed": status == "sampling_complete",
                    "sampler_diagnostics": {"rhat": np.array([np.nan]), "ess": np.array([8.])}}
        runs = [[result("sampling_complete", 1.)], [result("sampling_warning", 3.)],
                [result("sampling_warning", 7., False)], [result("failed", 5., False)]]
        estimates, statuses, _, _ = terminal_arrays(runs, ["BI_HMC"], 1)
        np.testing.assert_allclose(estimates[0, :2, 0], [1., 3.])
        self.assertTrue(np.isnan(estimates[0, 2:, 0]).all())
        finite = bias_statistics(estimates, statuses, [0.], "finite")
        passed = bias_statistics(estimates, statuses, [0.], "converged")
        self.assertEqual(finite["bias"][0, 0], 2.)
        self.assertEqual(passed["bias"][0, 0], 1.)
        self.assertEqual(finite["sampling_warning_count"][0], 2)
        self.assertEqual(finite["failed_count"][0], 1)
        details = terminal_metadata(runs, ["BI_HMC"])
        np.testing.assert_array_equal(details["sampler_diagnostics_passed"], [[True, False, False, False]])
        np.testing.assert_array_equal(details["point_estimate_valid"], [[True, True, False, False]])
        np.testing.assert_array_equal(details["total_wall_time"], [[3., 3., 3., 3.]])
        self.assertEqual(json.loads(details["sampler_diagnostics_json"][0, 0])["rhat"], [None])

    def test_progress_names_hmc_transitions_and_resets_for_the_next_optimizer(self):
        clock, console = [0.], io.StringIO()
        with redirect_stdout(console):
            progress = StudyProgress("test", ["BI_HMC", "WS-GGI"], 1, [10], 1,
                                     {"progress_interval_seconds": 1}, clock=lambda: clock[0])
            common = {"method": "BI_HMC", "record_length": 10, "run_metadata": {},
                      "method_kind": "bayesian", "max_transitions": 12}
            progress({**common, "phase": "method_start"})
            clock[0] = 2.
            progress({**common, "phase": "sampling", "transition": 3, "chain": 1,
                      "sampling_phase": "warmup"})
            clock[0] = 3.
            progress({**common, "phase": "method_end", "transition": 12, "status": "sampling_warning"})
            optimizer = {"method": "WS-GGI", "record_length": 10, "run_metadata": {}, "max_updates": 2}
            progress({**optimizer, "phase": "method_start"})
            clock[0] = 4.
            progress({**optimizer, "phase": "method_end", "iteration": 2, "status": "converged"})
        text = console.getvalue()
        self.assertIn("HMC transitions 3/12", text)
        self.assertIn("12 HMC transitions", text)
        self.assertNotIn("12 accepted updates", text)
        self.assertIn("starting; update budget 2", text)
        self.assertIn("2 accepted updates", text)


if __name__ == "__main__":
    unittest.main()
