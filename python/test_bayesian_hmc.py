"""Independent numerical checks for the BI_HMC likelihood, transforms and sampler."""
import unittest

import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov, toeplitz
from scipy.signal import lfilter

from bayesian_hmc import (BayesianWienerARModel, coefficients_to_coordinates,
                          denominator_log_jacobian, exact_lfilter, posterior_diagnostics,
                          reflection_to_coefficients, resolve_hmc_config, run_bi_hmc)
from wiener_system import WienerDimensions


class BayesianHMCTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(9876)

    def tearDown(self):
        if torch.get_num_threads() != self.previous_threads:
            torch.set_num_threads(self.previous_threads)

    def randn(self, length):
        return torch.randn(length, generator=self.generator, dtype=torch.float64)

    def test_exact_filter_forward_and_finite_record_adjoint(self):
        x = self.randn(15).requires_grad_()
        a = torch.tensor([-.2, .08], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([.1, -.4, .7], dtype=torch.float64, requires_grad=True)
        expected = lfilter(b.detach().numpy(), np.r_[1., a.detach().numpy()], x.detach().numpy())
        np.testing.assert_allclose(exact_lfilter(x, a, b).detach(), expected, atol=1e-13)
        self.assertTrue(torch.autograd.gradcheck(exact_lfilter, (x, a, b), atol=2e-6, rtol=1e-4))

    def test_stability_inverse_and_full_jacobian_for_arbitrary_orders(self):
        for order in range(6):
            z = (.3*self.randn(order)).requires_grad_()
            coefficients = reflection_to_coefficients(z)
            np.testing.assert_allclose(coefficients_to_coordinates(coefficients).numpy(), z.detach().numpy(), atol=1e-12)
            if order:
                self.assertLess(np.max(np.abs(np.roots(np.r_[1., coefficients.detach().numpy()]))), 1)
                jacobian = torch.autograd.functional.jacobian(reflection_to_coefficients, z)
                self.assertAlmostEqual(float(torch.linalg.slogdet(jacobian).logabsdet),
                                       float(denominator_log_jacobian(z)), places=10)
            else:
                self.assertEqual(float(denominator_log_jacobian(z)), 0)
        with self.assertRaises(ValueError):
            coefficients_to_coordinates(torch.tensor([1.1], dtype=torch.float64))

    def make_model(self, policy="stationary", burn=None):
        dims = WienerDimensions(na=2, nb=3, nf=3, nd=2)
        cfg = resolve_hmc_config({"noise_initialization": policy})
        r, c, history = self.randn(18), self.randn(18), self.randn(7)
        model = BayesianWienerARModel(r, c, dims, cfg, input_history=history, noise_burn_in=burn)
        z = .2*self.randn(dims.n_params+1)
        z[-1] = -1.3
        return model, z

    def test_observed_input_prehistory_and_exact_signal(self):
        model, z = self.make_model()
        theta, _, _ = model.transform(z)
        a, b = theta[:2].numpy(), theta[2:5].numpy()
        expected = lfilter(np.r_[0., b], np.r_[1., a], model.full_input.numpy())[model.input_count:]
        alpha, _ = model.signal(theta)
        np.testing.assert_allclose(alpha.detach().numpy(), expected, atol=1e-13)

    def test_stationary_and_finite_burn_likelihood_match_dense_covariance(self):
        for burn in (None, 0, 5):
            model, z = self.make_model(burn=burn)
            theta, _, _ = model.transform(z)
            _, beta = model.signal(theta)
            e = (model.c-beta).detach().numpy()
            d = theta[-2:].detach().numpy()
            length = len(e)
            if burn is None:
                transition = np.array([[-d[0], -d[1]], [1., 0.]])
                covariance = solve_discrete_lyapunov(transition, np.diag([1., 0.]))
                state = covariance[:, 0]
                gamma = []
                for lag in range(length):
                    gamma.append(state[0])
                    state = transition @ state
                matrix = toeplitz(gamma)
            else:
                impulse = np.zeros(length+burn)
                impulse[0] = 1
                response = lfilter([1.], np.r_[1., d], impulse)
                H = np.zeros((length, length+burn))
                for row in range(length):
                    time = burn+row
                    H[row, :time+1] = response[:time+1][::-1]
                matrix = H @ H.T
            q, logdet, count = model.likelihood_terms(theta)
            self.assertAlmostEqual(float(q), e @ np.linalg.solve(matrix, e), places=9)
            self.assertAlmostEqual(float(logdet), np.linalg.slogdet(matrix)[1], places=9)
            self.assertEqual(count, length)

    def test_log_posterior_gradients_all_boundary_policies(self):
        for policy, burn in (("stationary", None), ("conditional", None), ("zero", None), ("stationary", 5)):
            model, position = self.make_model(policy=policy, burn=burn)
            self.assertTrue(torch.autograd.gradcheck(model.log_prob, (position.requires_grad_(),),
                                                    atol=2e-5, rtol=2e-4))

    def test_short_record_ar_covariance_and_white_noise(self):
        for noise_order in (0, 4):
            dims = WienerDimensions(0, 1, 1, noise_order)
            model = BayesianWienerARModel(self.randn(2), self.randn(2), dims, resolve_hmc_config({}))
            z = .1*self.randn(dims.n_params+1)
            self.assertTrue(torch.isfinite(model.log_prob(z)))
            self.assertTrue(torch.autograd.gradcheck(model.log_prob, (z.requires_grad_(),), atol=1e-5))

    def test_unrepresentable_transform_proposals_have_zero_density(self):
        model, position = self.make_model()
        bad_variance = position.clone()
        bad_variance[-1] = 1000.
        self.assertTrue(torch.isneginf(model.log_prob(bad_variance)))
        rounded_boundary = position.clone()
        rounded_boundary[0] = 1000.
        self.assertTrue(torch.isneginf(model.log_prob(rounded_boundary)))

    def test_rng_isolation_reproducibility_and_fixed_retained_metric(self):
        dims = WienerDimensions(0, 1, 1, 0)
        r, c = self.randn(16), self.randn(16)
        config = {"chains": 2, "warmup": 25, "samples": 10, "leapfrog_steps": 3,
                  "initial_optimization_steps": 0, "progress_every": 1, "seed": 166}
        events = []
        before = torch.random.get_rng_state().clone()
        result = run_bi_hmc(r, c, dims, config, progress_callback=events.append)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        repeated = run_bi_hmc(r, c, dims, config)
        torch.testing.assert_close(result["posterior_samples"], repeated["posterior_samples"])
        for chain in (1, 2):
            retained = [e["step_size"] for e in events if e["chain"] == chain and e["sampling_phase"] == "sampling"]
            self.assertEqual(len(set(retained)), 1)
        self.assertEqual(result["posterior_samples"].shape, (2, 10, 1))
        self.assertEqual(result["sampling_transitions"], 70)
        self.assertGreater(result["total_time"], 0)
        self.assertEqual(result["point_estimator"], "posterior_mean")
        self.assertEqual(result["status"], "sampling_warning")

    def test_linear_case_posterior_matches_independent_quadrature(self):
        dims = WienerDimensions(0, 1, 1, 0)
        r = self.randn(40)
        x = torch.cat((r.new_zeros(1), r[:-1]))
        c = .7*x+.2*self.randn(40)
        config = {"chains": 2, "warmup": 220, "samples": 400, "leapfrog_steps": 7,
                  "initial_optimization_steps": 15, "coefficient_prior_std": 2.,
                  "variance_prior_shape": 3., "variance_prior_scale": .08,
                  "initial_variance": .04, "seed": 925, "min_ess": 20}
        result = run_bi_hmc(r, c, dims, config)
        grid = np.linspace(.2, 1.2, 20001)
        q = ((c.numpy()[:, None]-x.numpy()[:, None]*grid[None, :])**2).sum(axis=0)
        log_density = -.5*(grid/2)**2-(3+len(c)/2)*np.log(.08+q/2)
        weight = np.exp(log_density-log_density.max())
        weight /= weight.sum()
        expected_b = np.sum(grid*weight)
        expected_s = np.sum(weight*(.08+q/2)/(3+len(c)/2-1))
        self.assertAlmostEqual(float(result["theta_hat"][0]), expected_b, delta=.015)
        self.assertAlmostEqual(result["variance_hat"], expected_s, delta=.006)
        self.assertTrue(result["point_estimate_valid"])
        self.assertTrue(np.all(np.isfinite(result["sampler_diagnostics"]["rhat"])))

    def test_validation_and_diagnostics_screen_short_runs(self):
        for bad in ({"chains": 0}, {"warmup": -1}, {"target_accept": 1.}, {"initial_variance": 0.}):
            with self.assertRaises(ValueError):
                resolve_hmc_config(bad)
        diagnostic = posterior_diagnostics(np.ones((1, 4, 2)))
        self.assertTrue(np.isnan(diagnostic["rhat"]).all())

    def test_constant_continuous_parameter_chains_do_not_claim_convergence(self):
        diagnostic = posterior_diagnostics(np.ones((4, 100, 2)))
        self.assertTrue(np.isnan(diagnostic["rhat"]).all())
        self.assertTrue(np.isnan(diagnostic["bulk_ess"]).all())
        self.assertTrue(np.isnan(diagnostic["mean_ess"]).all())
        sign_flips = np.tile(np.array([1., -1.]), 50)[None, :, None].repeat(4, axis=0)
        self.assertTrue(np.isnan(posterior_diagnostics(sign_flips)["rhat"]).all())

    def test_cpu_thread_setting_is_restored_after_failure(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with self.assertRaises(ValueError):
                run_bi_hmc(self.randn(5), self.randn(5), WienerDimensions(0, 0, 1, 0),
                           {"cpu_threads": 1})
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
