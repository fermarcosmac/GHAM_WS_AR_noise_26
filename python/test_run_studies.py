"""Command ordering, backend dispatch, and zero-state Monte Carlo replay."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scipy.io import loadmat

import run_studies


class StudyLauncherTests(unittest.TestCase):
    def plan(self, *arguments):
        with redirect_stdout(io.StringIO()):
            return run_studies.main([*arguments, '--dry-run'])

    def test_paired_comparisons_run_fsm_export_then_matlab_for_each_benchmark(self):
        commands = self.plan('--study', 'montecarlo', '--cpu', '--runs', '2')
        self.assertEqual(len(commands), 6)
        for offset, benchmark in ((0, 'example_1'), (3, 'example_CSTR')):
            self.assertTrue(commands[offset][1].endswith('comparison_torch.py'))
            self.assertIn(benchmark, commands[offset])
            self.assertIn('--mode', commands[offset])
            self.assertIn('--cpu', commands[offset])
            self.assertTrue(commands[offset+1][1].endswith('export_shared_python_records.py'))
            self.assertIn('MONTECARLO', commands[offset+1])
            matlab = commands[offset+2]
            self.assertEqual(matlab[:2], ['matlab', '-batch'])
            self.assertIn('cfg.N_MC=2;', matlab[2])
            self.assertIn('python_record_library', matlab[2])

    def test_initialization_replays_the_matching_python_starts(self):
        commands = self.plan('--study', 'initialization', '--experiment', 'example_1')
        self.assertEqual(len(commands), 3)
        self.assertTrue(commands[0][1].endswith('initialization_study.py'))
        self.assertIn('INITIALIZATION', commands[1])
        self.assertIn('initialization_study(', commands[2][2])
        self.assertIn('shared_python_initialization.mat', commands[2][2])

    def test_noise_and_record_length_use_the_shared_configuration(self):
        for study, script in [('noise', 'noise_study.py'), ('record-length', 'record_length_study.py')]:
            commands = self.plan('--study', study, '--experiment', 'example_CSTR', '--iterations', '4')
            self.assertEqual(len(commands), 2)
            self.assertTrue(commands[0][1].endswith(script))
            self.assertIn('--study-config', commands[0])
            self.assertIn('--iterations', commands[0])
            self.assertIn('study_config', commands[1][2])
            self.assertIn('"K_max": 4', commands[1][2])
            self.assertNotIn('python_record_library', commands[1][2])

    def test_backend_selection_and_dry_run_do_not_launch_anything(self):
        with patch('run_studies.subprocess.run') as execute, patch('run_studies.shutil.which') as find:
            fsm = self.plan('--study', 'noise', '--parametrization', 'fsm')
            pl = self.plan('--study', 'noise', '--parametrization', 'pl')
        execute.assert_not_called()
        find.assert_not_called()
        self.assertEqual(len(fsm), 2)
        self.assertTrue(all(command[1].endswith('noise_study.py') for command in fsm))
        self.assertEqual(len(pl), 2)
        self.assertTrue(all(command[:2] == ['matlab', '-batch'] for command in pl))
        self.assertEqual(run_studies.matlab_string("O'Brien"), "'O''Brien'")

    def test_children_run_from_repo_root_and_stop_on_failure(self):
        with patch('run_studies.subprocess.run') as execute, redirect_stdout(io.StringIO()):
            commands = run_studies.main(['--study', 'noise', '--parametrization', 'fsm', '--experiment', 'example_1'])
        execute.assert_called_once_with(commands[0], cwd=run_studies.ROOT, check=True)

    def test_real_zero_state_montecarlo_output_replays_for_pl(self):
        from comparison_torch import main
        from export_shared_python_records import export_records
        from diagnostics import array_digest
        import numpy as np

        with tempfile.TemporaryDirectory(prefix='zero_state_mc_', dir=run_studies.ROOT/'python') as folder:
            folder = Path(folder)
            hmc = folder/'hmc.json'
            hmc.write_text(json.dumps(dict(chains=2, warmup=4, samples=8, leapfrog_steps=2,
                initial_optimization_steps=0, step_size=.001, adapt_mass=False, adapt_step_size=False)))
            output = folder/'comparison_fsm_montecarlo.mat'
            with redirect_stdout(io.StringIO()):
                main(['--experiment', 'example_1', '--mode', 'MONTECARLO', '--runs', '2',
                      '--iterations', '2', '--length', '24', '--methods', 'WS-GGI', 'BI_HMC',
                      '--cpu', '--no-show', '--hmc-config', str(hmc), '--output', str(output)])
                library = export_records(output, 'MONTECARLO')
            saved = loadmat(output, simplify_cells=True)
            bank = loadmat(library, simplify_cells=True)
            self.assertEqual(bank['r'].shape, (24, 2))
            self.assertEqual(saved['experiment']['initial_condition_policy'], 'zero_state_fit_after_burn_in')
            for i, run in enumerate(saved['all_runs']):
                bi = next(r for r in np.atleast_1d(run) if getattr(r, 'name', None) == 'BI_HMC')
                metadata = bi.run_metadata
                self.assertEqual(metadata.noise_initialization, 'zero')
                self.assertEqual(metadata.input_history_length, 0)
                self.assertEqual(metadata.input_history_sha256, array_digest(bank['input_prehistory'][:, i]))


if __name__ == '__main__':
    unittest.main()
