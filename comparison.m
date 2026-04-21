%% Wiener System Identification - Method Comparison Harness
%
% This script compares multiple identification methods on the Wiener system
% with AR noise from Lv et al. (2026). Hyperparameters are loaded from
% optim_configs.json so new methods can be added without changing the main
% experiment loop.

clear; clc; close all;
rng(42);

%% 0. Experiment configuration
config_file = 'optim_configs.json';

% List the methods to compare here. Unknown methods or placeholder methods
% are reported and skipped gracefully.
selected_methods = { ...
    'WS-GGI', ...
    'RGLS', ...
    'WS-GNI', ...
    'WS-GGHAM-1', ...
    'WS-GGHAM-2', ...
    'WS-LGHAM-1', ...
    'WS-LGHAM-2', ...
    'WS-LGHAM-3' ...
};

%% 1. True system parameters
na = 2;
nb = 2;
nf = 2;
nd = 1;
n = na + nb + (nf - 1) + nd;

a_true = [-0.31; -0.27];
b_true = [0.23; 0.98];
f_true = 0.32;
d_true = -0.40;

theta_true = [a_true; b_true; f_true; d_true];
param_names = build_param_names(na, nb, nf, nd);
param_labels = build_plot_labels(na, nb, nf, nd);

%% 2. Shared experiment settings
lambda_g = 1000;
K_max = 240;
conv_threshold = 1e-8;
sigma_nu = 0.10;
burn_in = 100;

%% 3. Generate input / noise / true output
N_total = lambda_g + burn_in;
r_full = randn(N_total, 1);
nu_full = sigma_nu * randn(N_total, 1);

poly_coeffs_true = [flipud(f_true(:)); 1; 0];

den_lin_true = [1; a_true(:)];
num_lin_true = [0; b_true(:)];
alpha_full = filter(num_lin_true.', den_lin_true.', r_full);

den_ar_true = [1; d_true(:)];
e_full = filter(1, den_ar_true.', nu_full);

beta_full = polyval(poly_coeffs_true.', alpha_full);
c_full = beta_full + e_full;

keep_idx = (burn_in + 1):(burn_in + lambda_g);
r = r_full(keep_idx);
nu = nu_full(keep_idx);
c = c_full(keep_idx);

%% 4. Load optimization configs
method_configs = load_method_configs(config_file);

%% 5. Run selected methods
results = [];
for m = 1:numel(selected_methods)
    method_name = selected_methods{m};
    method_cfg = get_method_config(method_configs, method_name);
    fprintf('\n%s\n', repmat('=', 1, 72));
    fprintf('Running method: %s\n', method_name);
    fprintf('%s\n', repmat('=', 1, 72));

    method_result = run_identification_method( ...
        method_name, method_cfg, r, nu, c, theta_true, ...
        na, nb, nf, nd, lambda_g, K_max, conv_threshold);

    if isempty(results)
        results = method_result;
    else
        results(end + 1) = method_result;
    end
end

%% 6. Summary table
fprintf('\n%s\n', repmat('=', 1, 92));
fprintf('FINAL METHOD SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 92));
fprintf('%-14s %-12s %-12s %-12s %-12s\n', ...
    'Method', 'Status', 'Final err(%)', 'Final d1', 'CPU time (s)');
fprintf('%s\n', repmat('-', 1, 92));
for m = 1:numel(results)
    final_d = NaN;
    if ~isempty(results(m).theta_hat)
        final_d = results(m).theta_hat(end);
    end
    fprintf('%-14s %-12s %-12.5f %-12.5f %-12.5f\n', ...
        results(m).name, results(m).status, results(m).final_err, ...
        final_d, results(m).total_time);
end

%% 7. Plots
plot_method_metrics(results);
plot_parameter_trajectories(results, theta_true, param_labels);

fprintf('\nDone. All figures generated.\n');


%% Local functions
function result = run_identification_method(method_name, method_cfg, r, nu, c, theta_true, na, nb, nf, nd, lambda_g, K_max, conv_threshold)
n = na + nb + (nf - 1) + nd;

alpha_hat = 1e-6 * ones(lambda_g, 1);
e_hat = 1e-6 * ones(lambda_g, 1);
nu_hat = 1e-6 * ones(lambda_g, 1);      % unnecessary to initialize
theta_hat = 1e-6 * ones(n, 1);

theta_hist = zeros(n, K_max);
param_err = NaN(K_max, 1);
rel_change = NaN(K_max, 1);
RMSE_hist = NaN(K_max, 1);
MAE_hist = NaN(K_max, 1);
iter_time = NaN(K_max, 1);
cum_time = NaN(K_max, 1);

status = 'ok';
status_msg = '';
iter_count = K_max;

for k = 1:K_max
    t_iter = tic;

    Phi_hat = build_state_matrix(alpha_hat, e_hat, r, na, nb, nf, nd);
    [theta_new, aux_state] = method_parameter_update(method_name, method_cfg, Phi_hat, c, theta_hat);

    if strcmp(aux_state.status, 'not_implemented')
        status = 'skipped';
        status_msg = aux_state.message;
        iter_count = max(k - 1, 0);
        break;
    end

    a_hat_v = theta_new(1:na);
    b_hat_v = theta_new(na + 1:na + nb);
    d_hat_v = theta_new(na + nb + (nf - 1) + 1:end);

    nu_hat = c - Phi_hat * theta_new;

    den_ar_hat = [1; d_hat_v(:)];
    e_hat = filter(1, den_ar_hat.', nu_hat);
    e_hat = e_hat(:);

    den_lin_hat = [1; a_hat_v(:)];
    num_lin_hat = [0; b_hat_v(:)];
    alpha_hat = filter(num_lin_hat.', den_lin_hat.', r);
    alpha_hat = alpha_hat(:);

    rel_change(k) = norm(theta_new - theta_hat) / (norm(theta_hat) + 1e-15);
    theta_hat = theta_new;
    theta_hist(:, k) = theta_hat;

    param_err(k) = norm(theta_hat - theta_true) / norm(theta_true) * 100;

    c_sim = simulate_wiener(r, nu, theta_hat, na, nb, nf, nd, lambda_g);
    res = c - c_sim;
    RMSE_hist(k) = sqrt(mean(res.^2));
    MAE_hist(k) = mean(abs(res));

    iter_time(k) = toc(t_iter);
    if k == 1
        cum_time(k) = iter_time(k);
    else
        cum_time(k) = cum_time(k - 1) + iter_time(k);
    end

    if ismember(k, [5 10 15 30 50 70 100 240]) || k == K_max
        fprintf('%-6d  ', k);
        fprintf('%-10.5f  ', theta_hat);
        fprintf('%-10.5f  %-10.5f\n', param_err(k), cum_time(k));
    end

    if k > 1 && rel_change(k) < conv_threshold
        iter_count = k;
        fprintf('*** %s converged at iteration %d ***\n', method_name, k);
        break;
    end
end

if strcmp(status, 'ok')
    theta_hist = theta_hist(:, 1:iter_count);
    param_err = param_err(1:iter_count);
    rel_change = rel_change(1:iter_count);
    RMSE_hist = RMSE_hist(1:iter_count);
    MAE_hist = MAE_hist(1:iter_count);
    iter_time = iter_time(1:iter_count);
    cum_time = cum_time(1:iter_count);
    final_err = param_err(end);
    total_time = cum_time(end);
else
    theta_hist = zeros(n, 0);
    param_err = zeros(0, 1);
    rel_change = zeros(0, 1);
    RMSE_hist = zeros(0, 1);
    MAE_hist = zeros(0, 1);
    iter_time = zeros(0, 1);
    cum_time = zeros(0, 1);
    final_err = NaN;
    total_time = 0;
    theta_hat = [];
end

if ~isempty(status_msg)
    fprintf('  %s\n', status_msg);
end

result = struct( ...
    'name', method_name, ...
    'status', status, ...
    'status_msg', status_msg, ...
    'theta_hat', theta_hat, ...
    'theta_hist', theta_hist, ...
    'param_err', param_err, ...
    'rel_change', rel_change, ...
    'RMSE_hist', RMSE_hist, ...
    'MAE_hist', MAE_hist, ...
    'iter_time', iter_time, ...
    'cum_time', cum_time, ...
    'final_err', final_err, ...
    'total_time', total_time, ...
    'iterations', iter_count);
end

function [theta_new, aux_state] = method_parameter_update(method_name, method_cfg, Phi_hat, c, theta_hat)
theta_new = theta_hat;
aux_state = struct('status', 'ok', 'message', '');

switch upper(method_name)
    case 'WS-GGI'
        if ~isfield(method_cfg, 'step_scale')
            error('WS-GGI config requires field "step_scale".');
        end
        delta = method_cfg.step_scale / max(norm(Phi_hat, 'fro')^2, eps);
        theta_new = theta_hat + delta * (Phi_hat' * (c - Phi_hat * theta_hat));

    case 'RGLS'
        required_fields = {'forgetting_factor', 'p0_scale'};
        assert_config_fields(method_cfg, required_fields, method_name);

        lambda_rls = method_cfg.forgetting_factor;
        P = method_cfg.p0_scale * eye(size(Phi_hat, 2));
        theta_rls = theta_hat;
        for t = 1:size(Phi_hat, 1)
            phi_t = Phi_hat(t, :).';
            gain_denom = lambda_rls + phi_t' * P * phi_t;
            gain_denom = max(real(gain_denom), eps);
            K_t = (P * phi_t) / gain_denom;
            theta_rls = theta_rls + K_t * (c(t) - phi_t' * theta_rls);
            P = (P - K_t * phi_t' * P) / lambda_rls;
            P = 0.5 * (P + P.');
        end
        theta_new = project_stable_theta(theta_rls, method_cfg);

    case {'WS-GNI', 'WS-GGHAM-1', 'WS-GGHAM-2', 'WS-LGHAM-1', 'WS-LGHAM-2', 'WS-LGHAM-3'}
        aux_state.status = 'not_implemented';
        aux_state.message = sprintf('%s placeholder selected. Add its theta-update rule in method_parameter_update().', method_name);

    otherwise
        aux_state.status = 'not_implemented';
        aux_state.message = sprintf('Unknown method "%s". Add its config and update rule before running it.', method_name);
end
end

function Phi_hat = build_state_matrix(alpha_hat, e_hat, r, na, nb, nf, nd)
T = numel(r);

alpha_lags = build_lag_matrix(alpha_hat, na);
r_lags = build_lag_matrix(r, nb);
e_lags = build_lag_matrix(e_hat, nd);

if nf > 1
    nonlinear_terms = zeros(T, nf - 1);
    for p = 2:nf
        nonlinear_terms(:, p - 1) = alpha_hat(:).^p;
    end
else
    nonlinear_terms = zeros(T, 0);
end

Phi_hat = [-alpha_lags, r_lags, nonlinear_terms, -e_lags];
end

function Xlag = build_lag_matrix(x, max_lag)
T = numel(x);
Xlag = zeros(T, max_lag);
x = x(:);

for lag = 1:max_lag
    Xlag((lag + 1):end, lag) = x(1:(end - lag));
end
end

function c_sim = simulate_wiener(r, nu, theta, na, nb, nf, nd, T)
expected_n = na + nb + (nf - 1) + nd;
assert(numel(theta) == expected_n, 'theta has wrong dimension.');

a_v = theta(1:na);
b_v = theta(na + 1:na + nb);
f_v = theta(na + nb + 1:na + nb + (nf - 1));
d_v = theta(na + nb + (nf - 1) + 1:end);

den_lin = [1; a_v(:)];
num_lin = [0; b_v(:)];
alpha = filter(num_lin.', den_lin.', r);

poly_coeffs = [flipud(f_v(:)); 1; 0];
beta = polyval(poly_coeffs.', alpha);

den_ar = [1; d_v(:)];
e = filter(1, den_ar.', nu);

c_sim = beta(:) + e(:);

if nargin >= 8
    c_sim = c_sim(1:T);
end
end

function theta_proj = project_stable_theta(theta, method_cfg)
theta_proj = theta;

if ~isfield(method_cfg, 'stability_projection') || ~method_cfg.stability_projection
    return;
end

required_fields = {'na', 'nb', 'nf', 'nd', 'stability_radius'};
assert_config_fields(method_cfg, required_fields, 'RGLS');

na = method_cfg.na;
nb = method_cfg.nb;
nf = method_cfg.nf;
nd = method_cfg.nd;
radius = method_cfg.stability_radius;

a_idx = 1:na;
d_idx = na + nb + (nf - 1) + 1 : na + nb + (nf - 1) + nd;

theta_proj(a_idx) = stabilize_denominator_coeffs(theta_proj(a_idx), radius);
theta_proj(d_idx) = stabilize_denominator_coeffs(theta_proj(d_idx), radius);
end

function coeffs_stable = stabilize_denominator_coeffs(coeffs, radius)
den_poly = [1; coeffs(:)];
pole_locations = roots(den_poly.');

for i = 1:numel(pole_locations)
    if abs(pole_locations(i)) >= radius
        pole_locations(i) = radius * pole_locations(i) / abs(pole_locations(i));
    end
end

den_poly_stable = poly(pole_locations);
den_poly_stable = real(den_poly_stable(:));
coeffs_stable = den_poly_stable(2:end);
end

function configs = load_method_configs(config_file)
assert(exist(config_file, 'file') == 2, 'Could not find config file: %s', config_file);
raw_text = fileread(config_file);
decoded = jsondecode(raw_text);
assert(isfield(decoded, 'methods'), 'JSON config must contain a "methods" field.');
configs = decoded.methods;
end

function method_cfg = get_method_config(configs, method_name)
field_name = matlab.lang.makeValidName(method_name);
assert(isfield(configs, field_name), 'No config entry found for method "%s".', method_name);
method_cfg = configs.(field_name);
end

function assert_config_fields(cfg, required_fields, method_name)
for i = 1:numel(required_fields)
    assert(isfield(cfg, required_fields{i}), ...
        '%s config requires field "%s".', method_name, required_fields{i});
end
end

function names = build_param_names(na, nb, nf, nd)
names = [arrayfun(@(i) sprintf('a%d', i), 1:na, 'UniformOutput', false), ...
         arrayfun(@(i) sprintf('b%d', i), 1:nb, 'UniformOutput', false), ...
         arrayfun(@(i) sprintf('f%d', i), 2:nf, 'UniformOutput', false), ...
         arrayfun(@(i) sprintf('d%d', i), 1:nd, 'UniformOutput', false)];
end

function labels = build_plot_labels(na, nb, nf, nd)
labels = [arrayfun(@(i) sprintf('a_%d', i), 1:na, 'UniformOutput', false), ...
          arrayfun(@(i) sprintf('b_%d', i), 1:nb, 'UniformOutput', false), ...
          arrayfun(@(i) sprintf('f_%d', i), 2:nf, 'UniformOutput', false), ...
          arrayfun(@(i) sprintf('d_%d', i), 1:nd, 'UniformOutput', false)];
end

function plot_method_metrics(results)
valid_idx = find(strcmp({results.status}, 'ok'));
if isempty(valid_idx)
    warning('No implemented methods produced results. Skipping plots.');
    return;
end

colors = lines(numel(valid_idx));

figure('Name', 'Method Comparison Metrics', 'Color', 'w', 'Position', [80 80 1200 800]);

subplot(2, 2, 1); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    plot(1:results(idx).iterations, results(idx).param_err, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
grid on; xlabel('Iteration'); ylabel('err (%)');
title('Parameter Error vs Iteration'); legend('Location', 'best');

subplot(2, 2, 2); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    plot(results(idx).cum_time, results(idx).param_err, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
grid on; xlabel('Compute time (s)'); ylabel('err (%)');
title('Parameter Error vs Compute Time'); legend('Location', 'best');

subplot(2, 2, 3); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    plot(1:results(idx).iterations, results(idx).RMSE_hist, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
grid on; xlabel('Iteration'); ylabel('RMSE');
title('RMSE vs Iteration'); legend('Location', 'best');

subplot(2, 2, 4); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    plot(results(idx).cum_time, results(idx).RMSE_hist, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
grid on; xlabel('Compute time (s)'); ylabel('RMSE');
title('RMSE vs Compute Time'); legend('Location', 'best');

sgtitle('Wiener System Identification - Method Comparison', 'FontSize', 13, 'FontWeight', 'bold');

figure('Name', 'MAE Comparison', 'Color', 'w', 'Position', [120 120 1200 500]);
subplot(1, 2, 1); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    plot(1:results(idx).iterations, results(idx).MAE_hist, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
grid on; xlabel('Iteration'); ylabel('MAE');
title('MAE vs Iteration'); legend('Location', 'best');

subplot(1, 2, 2); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    plot(results(idx).cum_time, results(idx).MAE_hist, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
grid on; xlabel('Compute time (s)'); ylabel('MAE');
title('MAE vs Compute Time'); legend('Location', 'best');
end

function plot_parameter_trajectories(results, theta_true, param_labels)
valid_idx = find(strcmp({results.status}, 'ok'));
if isempty(valid_idx)
    return;
end

n = numel(theta_true);
colors = lines(numel(valid_idx));
figure('Name', 'Parameter Trajectories by Method', 'Color', 'w', 'Position', [150 100 1200 700]);

for i = 1:n
    subplot(ceil(n / 3), 3, i); hold on;
    for j = 1:numel(valid_idx)
        idx = valid_idx(j);
        plot(1:results(idx).iterations, results(idx).theta_hist(i, :), ...
            'LineWidth', 1.4, 'Color', colors(j, :), 'DisplayName', results(idx).name);
    end
    yline(theta_true(i), 'k--', 'LineWidth', 1.1, 'DisplayName', 'True');
    grid on;
    xlabel('Iteration');
    ylabel(['$\hat{', param_labels{i}, '}$'], 'Interpreter', 'latex');
    title(['Parameter ', param_labels{i}]);
    legend('Location', 'best');
end

sgtitle('Parameter Trajectories Across Methods', 'FontSize', 12, 'FontWeight', 'bold');
end
