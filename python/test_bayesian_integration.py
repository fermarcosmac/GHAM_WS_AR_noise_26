"""Bayesian dispatch, non-oracle inference, mixed export, and final-score semantics."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import loadmat

from common import load_json_config, set_seed
from comparison_torch import (build_dims_and_theta, inference_data_context,
    load_experiment_config, run_identification_method, run_method_set,
    save_comparison_results, select_python_methods, summarize_montecarlo)
from bayesian_plots import plot_bayesian_posteriors, plot_bayesian_outputs
import matplotlib.pyplot as plt


class BayesianIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_experiment_config("example_1")
        cls.cfg["settings"].update(lambda_g=28, burn_in=5, K_max=3)
        cls.cfg["initialization"] = {"policy": "consistent", "theta0": [0,0,1,0,0,0]}
        cls.cfg["diagnostics"] = {"enabled": False, "save_matrices": False}
        cls.dims, cls.truth = build_dims_and_theta(cls.cfg, dtype=torch.float64, device=torch.device("cpu"))
        cls.configs = load_json_config(Path(__file__).parent/"optim_configs.json")["methods"]
        cls.configs = copy.deepcopy(cls.configs)
        cls.configs["BI_HMC"].update(chains=2, warmup=5, samples=8, leapfrog_steps=2,
                                    initial_optimization_steps=4, progress_every=100, min_ess=1)
        set_seed(73)
        from wiener_system import generate_example_data
        cls.data = generate_example_data(cls.dims, cls.truth, 28, 5, .1,
                                        device=torch.device("cpu"), dtype=torch.float64)

    def run_methods(self, methods, *, truth=None, nu=None):
        d = self.data
        return run_method_set(methods, self.configs, d["r"], d["nu"] if nu is None else nu,
            d["c"], self.truth if truth is None else truth, self.dims, 28, 3, 0., False,
            verbose=False, experiment_cfg=self.cfg, run_metadata={"mc_index":1,"data_seed":73},
            data_context=inference_data_context(d, 5))

    def test_separate_loop_and_physical_posterior_mean(self):
        with patch("comparison_torch.method_parameter_update", side_effect=AssertionError("optimizer loop used")):
            result = self.run_methods(["BI_HMC"])[0]
        self.assertEqual(result["method_kind"], "bayesian")
        self.assertEqual(result["posterior_samples"].shape, (2,8,6))
        np.testing.assert_allclose(result["theta_hat"], result["posterior_samples"].mean(axis=(0,1)))
        np.testing.assert_allclose(result["final_err"],
            100*np.linalg.norm(result["theta_hat"]-self.truth.numpy())/np.linalg.norm(self.truth.numpy()))
        self.assertEqual(result["theta_hist"].shape, (6,0))
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["accepted_updates"], 0)
        self.assertEqual(result["sampling_transitions"], 26)
        self.assertGreaterEqual(result["total_wall_time"], result["inference_wall_time"])
        self.assertEqual(result["run_metadata"]["input_history_length"], 5)
        # Optimizer budgets cannot accidentally govern the HMC loop.
        direct = run_identification_method("BI_HMC", self.configs["BI_HMC"], self.data["r"],
            self.data["nu"], self.data["c"], self.truth, self.dims, 28, 0, -1.,
            verbose=False, data_context=inference_data_context(self.data,5))
        self.assertEqual(direct["posterior_samples"].shape, (2,8,6))

    def test_truth_and_oracle_innovations_change_scores_not_inference(self):
        first = self.run_methods(["BI_HMC"])[0]
        other = self.run_methods(["BI_HMC"], truth=self.truth+1., nu=self.data["nu"]+10.)[0]
        np.testing.assert_array_equal(first["posterior_samples"], other["posterior_samples"])
        np.testing.assert_array_equal(first["posterior_variance_samples"], other["posterior_variance_samples"])
        self.assertEqual(first["sampler_seed"], other["sampler_seed"])
        self.assertNotEqual(first["final_err"], other["final_err"])
        self.assertNotEqual(first["final_output_rmse"], other["final_output_rmse"])

    def test_mixed_method_export_preserves_bayesian_fields_in_either_order(self):
        results = self.run_methods(["WS-GGI","BI_HMC"])
        for ordered in [results, list(reversed(results))]:
            methods = [result["name"] for result in ordered]
            summary = summarize_montecarlo([ordered], methods, self.truth, 3)
            index = methods.index("BI_HMC")
            self.assertTrue(np.isnan(summary["param_err_runs"][index]).all())
            self.assertTrue(np.isnan(summary["rmse_runs"][index]).all())
            np.testing.assert_allclose(summary["final_param_estimates"][index,0], results[1]["theta_hat"])
            self.assertTrue(np.isfinite(summary["final_errors"][index,0]))
            self.assertTrue(np.isfinite(summary["final_output_rmse_runs"][index,0]))
            self.assertGreater(summary["total_wall_time_runs"][index,0], 0)
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
                path = Path(temporary)/"mixed.mat"
                save_comparison_results(path, ordered, experiment_name="test", mode="EXAMPLE",
                    selected_methods=methods, dims=self.dims, theta_true=self.truth, lambda_g=28,
                    K_max=3, conv_threshold=0., sigma_nu=.1, burn_in=5,
                    enforce_fixed_iterations=False, seed=73, mc_summary=summary, all_runs=[ordered])
                saved = loadmat(path, simplify_cells=True)
            self.assertEqual(saved["results"][index]["posterior_samples"].shape, (2,8,6))
            self.assertEqual(saved["results"][index]["point_estimate"], "posterior_mean")

    def test_exact_method_selection_and_posterior_figures(self):
        cfg = {**self.cfg, "python_selected_methods":["BI_HMC"], "include_python_extra_methods":False}
        self.assertEqual(select_python_methods(cfg), ["BI_HMC"])
        result = self.run_methods(["BI_HMC"])[0]
        figures = [plot_bayesian_posteriors(result,self.truth.numpy(),["a_1","a_2","b_1","b_2","f_2","d_1"]),
                   plot_bayesian_outputs(result,self.data["c"].numpy())]
        self.assertTrue(all(figure is not None for figure in figures))
        for figure in figures:
            figure.canvas.draw()
            plt.close(figure)

    def test_nested_sampler_overrides_control_actual_seed_and_budget(self):
        config = copy.deepcopy(self.configs["BI_HMC"])
        config["hmc"] = {"seed": 9071, "chains": 2, "warmup": 2, "samples": 8,
                         "leapfrog_steps": 1, "initial_optimization_steps": 0}
        events = []
        d = self.data
        result = run_method_set(["BI_HMC"], {"BI_HMC": config}, d["r"], d["nu"], d["c"],
            self.truth, self.dims, 28, 3, 0., False, verbose=False,
            experiment_cfg=self.cfg, run_metadata={"mc_index": 1},
            data_context=inference_data_context(d, 5), progress_callback=events.append)[0]
        self.assertEqual(result["sampler_seed"], 9071)
        self.assertEqual(result["posterior_samples"].shape, (2,8,6))
        self.assertEqual(result["sampling_transitions"], 20)
        self.assertEqual(events[0]["max_transitions"], 20)


if __name__ == "__main__":
    unittest.main()
