"""Checks experimental isolation, dependent-data pairing, and signed bias statistics."""
import copy
import io
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from common import load_json_config
from comparison_torch import build_dims_and_theta, load_experiment_config
from study_progress import StudyProgress
from identification_studies import (bias_statistics, reference_data, run_initialization_study,
                                    run_record_length_study, sample_initializations, stable)


class IdentificationStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_experiment_config('example_1')
        cls.cfg['settings'].update(lambda_g=32, burn_in=4, K_max=3, conv_threshold=0)
        cls.cfg['diagnostics'] = {'enabled': False, 'save_matrices': False}
        cls.settings = load_json_config(Path(__file__).resolve().parents[1]/'configs'/'studies.json')
        cls.dims, cls.theta = build_dims_and_theta(cls.cfg, dtype=torch.float64, device=torch.device('cpu'))

    def test_progress_throttles_heartbeats_and_weights_eta_by_method_and_length(self):
        clock = [0.]
        console = io.StringIO()
        with redirect_stdout(console):
            progress = StudyProgress('FSM record_length', ['fast', 'slow'], 1, [10, 20], 1,
                                     {'progress_interval_seconds': 5}, clock=lambda: clock[0])
            def event(phase, method, iteration=0, status=''):
                progress({'phase': phase, 'method': method, 'record_length': 10,
                          'max_updates': 9, 'run_metadata': {'mc_index': 1, 'resolution': 'test'},
                          'iteration': iteration, 'status': status})
            event('method_start', 'fast')
            clock[0] = 1.
            event('method_end', 'fast', status='skipped')
            event('method_start', 'slow')
            before = console.getvalue()
            clock[0] = 2.
            event('iteration', 'slow', 1)
            self.assertEqual(console.getvalue(), before)
            clock[0] = 6.
            event('iteration', 'slow', 4)
            clock[0] = 11.
            event('method_end', 'slow', 5, 'stalled')
        text = console.getvalue()
        self.assertIn('ETA warming up', text)
        self.assertIn('updates 4/9', text)
        self.assertIn('stalled; 5 accepted updates', text)
        self.assertIn('2/4 fits completed (50.0%)', text)
        # Remaining N=20 for each method, at 0.1 and 1 second/sample.
        self.assertIn('ETA ~00:00:22', text)
        with redirect_stdout(io.StringIO()) as quiet:
            disabled = StudyProgress('test', ['fast'], 1, [10], 1, {'progress': False})
            disabled.stage('Preparing data')
            disabled({'phase': 'method_start'})
            disabled.finish('output.mat')
        self.assertEqual(quiet.getvalue(), '')

    def test_starts_are_stable_reproducible_and_not_truth_centered(self):
        settings = copy.deepcopy(self.settings['initialization_study'])
        settings['n_runs'] = 12
        first = sample_initializations(self.dims, settings)
        np.testing.assert_array_equal(first, sample_initializations(self.dims, settings))
        self.assertEqual(np.unique(first, axis=1).shape[1], 12)
        self.assertTrue(all(stable(t, self.dims, .95) for t in first.T))
        settings['scale'] = 0
        with self.assertRaises(ValueError):
            sample_initializations(self.dims, settings)

    def test_bias_is_signed_mean_and_mcse_uses_replicate_count(self):
        values = np.array([[[2.], [4.], [np.nan]], [[9.], [np.nan], [np.nan]]])
        statuses = np.array([['converged', 'stalled', 'failed'],
                             ['converged', 'skipped', 'failed']], dtype=object)
        stats = bias_statistics(values, statuses, np.array([1.]))
        np.testing.assert_allclose(stats['bias'][:, 0], [2., 8.])
        self.assertAlmostEqual(stats['mcse'][0, 0], 1.)
        self.assertTrue(np.isnan(stats['mcse'][1, 0]))
        np.testing.assert_array_equal(stats['count'][:, 0], [2, 1])
        self.assertAlmostEqual(stats['rmse'][0, 0], np.sqrt(5))
        conditional = bias_statistics(values, statuses, [1.], 'converged')
        self.assertEqual(conditional['bias'][0, 0], 1.)

    def test_initialization_study_keeps_data_fixed_and_shares_each_start(self):
        settings = copy.deepcopy(self.settings['initialization_study'])
        settings['n_runs'] = 3
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            console = io.StringIO()
            with redirect_stdout(console):
                path = run_initialization_study(self.cfg, settings, output_file=Path(directory)/'init.mat')
            saved = loadmat(path, simplify_cells=True)
        self.assertIn('27/27 fits completed (100.0%)', console.getvalue())
        self.assertIn('initialization 3/3', console.getvalue())
        self.assertIn('skipped', console.getvalue())
        self.assertIn('Finished 27/27 method fits', console.getvalue())
        self.assertEqual(saved['study']['type'], 'initialization')
        starts = saved['study']['initializations']
        data_ids = set()
        init_ids = set()
        for i, run in enumerate(saved['all_runs']):
            for result in run:
                # SciPy versions differ in recursive conversion of nested cell structs.
                get = lambda obj, key: obj[key] if isinstance(obj, dict) else getattr(obj, key)
                np.testing.assert_array_equal(get(result, 'theta0'), starts[:, i])
                metadata = get(result, 'run_metadata')
                data_ids.add(get(metadata, 'dataset_id'))
                init_ids.add(get(metadata, 'initialization_sha256'))
        self.assertEqual(len(data_ids), 1)
        self.assertEqual(len(init_ids), 3)
        # A placeholder remains visible as skipped; no synthetic estimate is included.
        self.assertTrue(np.isnan(saved['mc_summary']['final_param_estimates'][0]).all())

    def test_record_length_study_pairs_prefixes_and_preserves_resolution(self):
        settings = copy.deepcopy(self.settings['record_length_study'])
        settings.update(n_runs=2, record_lengths=[16, 24, 32],
                        fsm_resolutions=[{'label': 'small', 'fsm_tau_g': 8, 'fsm_tau_h': 8, 'fsm_n_freq': 64},
                                         {'label': 'large', 'fsm_tau_g': 12, 'fsm_tau_h': 12, 'fsm_n_freq': 128}])
        cfg = copy.deepcopy(self.cfg)
        cfg['selected_methods'] = ['WS-GNI', 'WS-GGI']
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            console = io.StringIO()
            with redirect_stdout(console):
                path = run_record_length_study(cfg, settings, output_file=Path(directory)/'bias.mat')
            saved = loadmat(path, simplify_cells=True)['study']
        self.assertIn('48/48 fits completed (100.0%)', console.getvalue())
        self.assertIn('large; N=32; realization 2/2', console.getvalue())
        self.assertIn('ETA ~', console.getvalue())
        self.assertIn('Finished 48/48 method fits', console.getvalue())
        self.assertEqual(saved['final_param_estimates'].shape, (2, 3, 4, 2, 6))
        self.assertEqual(saved['bias'].shape, (2, 3, 4, 6))
        np.testing.assert_array_equal(saved['dataset_ids'][0], saved['dataset_ids'][1])
        self.assertNotEqual(saved['dataset_ids'][0, 0, 0, 0], saved['dataset_ids'][0, 0, 0, 1])
        raw = reference_data(cfg, self.dims, self.theta, 32, settings['data_seed'], torch.device('cpu'))
        from diagnostics import array_digest
        self.assertEqual(saved['dataset_ids'][0, 0, 0, 0], array_digest(raw['r'][:16], raw['c'][:16]))
        np.testing.assert_array_equal(saved['theta0'], [0, 0, 1, 0, 0, 0])
        self.assertTrue(np.isfinite(saved['inverse_n_coefficients']).all())
        settings['record_lengths'] = [4, 8]
        with self.assertRaises(ValueError):
            run_record_length_study(cfg, settings)


if __name__ == '__main__':
    unittest.main()
