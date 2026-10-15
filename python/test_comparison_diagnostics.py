"""Small numerical and export checks; never overwrite manuscript results.

Run: python -B -m unittest discover -s python -p 'test_*.py'
"""
import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import loadmat

from common import load_json_config, set_seed
from comparison_torch import (build_dims_and_theta, load_experiment_config,
    run_identification_method, run_method_set, save_comparison_results, summarize_montecarlo)
from diagnostics import matrix_metrics, resolve_method_config
from methods import _finalize_update
from wiener_system import generate_example_data, initialize_method_state, simulate_signal_path_fsm


class ComparisonDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = load_experiment_config('example_1')
        cls.dims, cls.theta = build_dims_and_theta(cls.exp, dtype=torch.float64, device=torch.device('cpu'))
        set_seed(42)
        cls.data = generate_example_data(cls.dims, cls.theta, 32, 4, .1,
                                        dtype=torch.float64, device=torch.device('cpu'))
        cls.configs = load_json_config(Path(__file__).parent / 'optim_configs.json')['methods']
        cls.initialization = {'policy': 'consistent', 'theta0': [0, 0, 1, 0, 0, 0]}

    def config(self, key='WS_GNI'):
        return resolve_method_config(self.configs[key], {'initialization': self.initialization})

    def run_method(self, name='WS-GNI', cfg=None, K=4, threshold=0):
        d = self.data
        return run_identification_method(name, cfg or self.config(), d['r'], d['nu'], d['c'],
            self.theta, self.dims, 32, K, threshold, enforce_fixed_iterations=True, verbose=False)

    def test_explicit_initialization_and_validation(self):
        cfg = self.config()
        state = initialize_method_state(cfg, self.data['r'], self.data['c'], self.dims)
        np.testing.assert_array_equal(state['theta_hat'].numpy(), self.initialization['theta0'])
        alpha, beta = simulate_signal_path_fsm(self.data['r'], state['theta_hat'], self.dims)
        torch.testing.assert_close(state['alpha_hat'], alpha)
        torch.testing.assert_close(state['e_hat'], self.data['c']-beta)
        for invalid in ([0, 1], [2, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, float('nan')]):
            cfg['initialization']['theta0'] = invalid
            with self.assertRaises(ValueError):
                initialize_method_state(cfg, self.data['r'], self.data['c'], self.dims)
        legacy = initialize_method_state(self.configs['WS_GNI'], self.data['r'], self.data['c'], self.dims)
        torch.testing.assert_close(legacy['theta_hat'], torch.full_like(self.theta, 1e-6))

    def test_singularity_is_not_missing_or_negative_curvature(self):
        singular = matrix_metrics(torch.diag(torch.tensor([2., 0.], dtype=torch.float64)))
        self.assertEqual(singular['condition'], float('inf'))
        self.assertEqual(singular['rank'], 1)
        indefinite = matrix_metrics(torch.diag(torch.tensor([-2., 1.], dtype=torch.float64)))
        self.assertEqual(indefinite['condition'], 2.)
        self.assertEqual(indefinite['negative_count'], 1)
        self.assertEqual(indefinite['singular'], 0)

    def test_diagnostics_do_not_change_updates_and_align_preupdate_state(self):
        cfg = self.config()
        cfg['diagnostics']['save_matrices'] = True
        result = self.run_method(cfg=cfg)
        off = copy.deepcopy(cfg)
        off['diagnostics']['enabled'] = False
        other = self.run_method(cfg=off)
        np.testing.assert_array_equal(result['theta_hist'], other['theta_hist'])
        np.testing.assert_array_equal(result['diagnostics']['evaluation_iteration'], [0, 1, 2])
        first = result['derivative_snapshots'][0]
        raw = matrix_metrics(torch.as_tensor(first['hess']))
        self.assertAlmostEqual(result['diagnostics']['raw_condition'][0], raw['condition'])
        np.testing.assert_allclose(first['hess_reg'], first['hess'] + cfg['hessian_regularization']*np.eye(6))
        self.assertNotIn('raw_condition', other['diagnostics'])
        self.assertAlmostEqual(result['total_wall_time'], result['total_time']+result['diagnostic_time'])
        self.assertEqual(result['run_metadata']['dataset_id'], other['run_metadata']['dataset_id'])
        self.assertNotEqual(result['run_metadata']['run_id'], other['run_metadata']['run_id'])

    def test_stopping_and_padding(self):
        cfg = self.config('WS_GGI')
        cfg['step_size'] = 0.
        stalled = self.run_method('WS-GGI', cfg, K=5, threshold=1e-8)
        self.assertEqual(stalled['status'], 'stalled')
        self.assertEqual(stalled['accepted_updates'], 1)
        np.testing.assert_array_equal(stalled['active_mask'], [True, True, False, False, False])
        self.assertEqual(len(stalled['diagnostics']['evaluation_iteration']), 1)
        self.assertNotIn('raw_condition', stalled['diagnostics'])
        cfg['stopping']['gradient_tolerance'] = 1e100
        converged = self.run_method('WS-GGI', cfg, threshold=1e-8)
        self.assertEqual(converged['status'], 'converged')
        initial = self.run_method(K=1)
        self.assertEqual(initial['accepted_updates'], 0)
        self.assertEqual(initial['theta_hist'].shape, (6, 1))

    def test_invalid_updates_cannot_be_reported_as_convergence(self):
        def invalid(*args, **kwargs):
            previous = args[4]
            return _finalize_update(torch.full_like(previous, float('nan')), previous,
                {'status': 'ok', 'message': '', 'derivatives': {}}, args[1], kwargs['dims'])
        with patch('comparison_torch.method_parameter_update', side_effect=invalid):
            result = self.run_method(threshold=1e-8)
        self.assertEqual(result['status'], 'invalid_update')
        self.assertEqual(result['accepted_updates'], 0)
        self.assertTrue(np.isnan(result['final_err']))
        self.assertEqual(result['diagnostic_summary']['rejected_count'], 1)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            summary = summarize_montecarlo([[result]], ['WS-GNI'], self.theta, 4)
        self.assertTrue(np.isnan(summary['final_param_estimates']).all())
        self.assertTrue(np.isnan(summary['param_err_runs']).all())
        np.testing.assert_array_equal(summary['param_err_count'], [[0, 0, 0, 0]])

    def test_montecarlo_counts_exclude_failed_runs(self):
        first = self.run_method()
        second, failed = copy.deepcopy(first), copy.deepcopy(first)
        first['param_err'][:] = 1
        second['param_err'][:] = 3
        failed['status'], failed['final_err'] = 'failed', np.nan
        summary = summarize_montecarlo([[first], [second], [failed]], ['WS-GNI'], self.theta, 4)
        np.testing.assert_array_equal(summary['param_err_count'], [[2, 2, 2, 2]])
        np.testing.assert_allclose(summary['param_err_mean'], 2.)
        np.testing.assert_allclose(summary['param_err_stderr'], 1.)
        self.assertTrue(np.isnan(summary['final_param_estimates'][0, 2]).all())

    def test_all_methods_and_matlab_export_preserve_every_run(self):
        d = self.data
        runs = run_method_set(self.exp['selected_methods'], self.configs, d['r'], d['nu'], d['c'],
            self.theta, self.dims, 32, 3, 0, True, verbose=False,
            experiment_cfg={'initialization': self.initialization}, run_metadata={'data_seed': 42, 'mc_index': 1})
        self.assertEqual(runs[0]['status'], 'skipped')
        self.assertTrue(all(r['status'] == 'max_iterations' for r in runs[1:]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/'results.mat'
            save_comparison_results(path, runs, experiment_name='test', mode='MONTECARLO',
                selected_methods=self.exp['selected_methods'], dims=self.dims, theta_true=self.theta,
                lambda_g=32, K_max=3, conv_threshold=0, sigma_nu=.1, burn_in=4,
                enforce_fixed_iterations=True, seed=42, n_mc=2, all_runs=[runs, runs], resolved_experiment=self.exp)
            saved = loadmat(path, simplify_cells=True)
            self.assertEqual(saved['schema_version'], 2)
            self.assertEqual(len(saved['all_runs']), 2)
            nested = saved['all_runs'][1][1]
            diagnostics = nested['diagnostics'] if isinstance(nested, dict) else nested.diagnostics
            values = diagnostics['raw_condition'] if isinstance(diagnostics, dict) else diagnostics.raw_condition
            self.assertEqual(values.size, 2)
            self.assertEqual(json.loads(saved['results'][1]['resolved_config_json'])['initialization'], self.initialization)


if __name__ == '__main__':
    unittest.main()
