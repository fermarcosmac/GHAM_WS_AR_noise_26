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
    'RGLS', ...
    'WS-GNI', ...
    'WS-GGI', ...
    'WS-GGHAM-1-dH', ...
    'WS-GGHAM-2-I', ...
    'WS-GGHAM-2-dH', ...
    'WS-LGHAM-1-TIK', ...
    'WS-LGHAM-2-TIK', ...
    'WS-LGHAM-3-TIK' ...
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

%na = 10;
%nb = 10;
%nf = 2;
%n = na + nb + (nf - 1) + nd;
%a_true = -1e-1*rand(na,1);
%b_true = 1e-1*rand(nb,1);
%f_true = randn(nf - 1, 1);

theta_true = [a_true; b_true; f_true; d_true];
param_names = build_param_names(na, nb, nf, nd);
param_labels = build_plot_labels(na, nb, nf, nd);

%% 2. Shared experiment settings
lambda_g = 1000;
K_max = 270*4;
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

init_state = initialize_method_state(method_cfg, r, c, na, nb, lambda_g, n);

alpha_hat = init_state.alpha_hat;
e_hat = init_state.e_hat;
nu_hat = 1e-6 * ones(lambda_g, 1);  % no need to initialize this
theta_hat = init_state.theta_hat;

theta_hist = zeros(n, K_max);
param_err = NaN(K_max, 1);
rel_change = NaN(K_max, 1);
RMSE_hist = NaN(K_max, 1);
iter_time = NaN(K_max, 1);
cum_time = NaN(K_max, 1);

status = 'ok';
status_msg = '';
iter_count = K_max;

% Record common pre-update metrics so all methods start from iteration 0.
theta_hist(:, 1) = theta_hat;
param_err(1) = norm(theta_hat - theta_true) / norm(theta_true) * 100;
c_sim_0 = simulate_wiener(r, nu, theta_hat, na, nb, nf, nd, lambda_g);
res_0 = c - c_sim_0;
RMSE_hist(1) = sqrt(mean(res_0.^2));
rel_change(1) = 0;
iter_time(1) = 0;
cum_time(1) = 0;

for k = 2:K_max
    t_iter = tic;

    Phi_hat = build_state_matrix(alpha_hat, e_hat, r, na, nb, nf, nd);
    [theta_new, aux_state] = method_parameter_update(method_name, method_cfg, Phi_hat, c, theta_hat, k - 1);

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

    iter_time(k) = toc(t_iter);
    cum_time(k) = cum_time(k - 1) + iter_time(k);

    iter_num = k - 1;
    if ismember(iter_num, [5 10 15 30 50 70 100 240]) || iter_num == (K_max - 1)
        fprintf('%-6d  ', iter_num);
        fprintf('%-10.5f  ', theta_hat);
        fprintf('%-10.5f  %-10.5f\n', param_err(k), cum_time(k));
    end

    if rel_change(k) < conv_threshold
        iter_count = k;
        fprintf('*** %s converged at iteration %d ***\n', method_name, iter_num);
        break;
    end
end

if strcmp(status, 'ok')
    theta_hist = theta_hist(:, 1:iter_count);
    param_err = param_err(1:iter_count);
    rel_change = rel_change(1:iter_count);
    RMSE_hist = RMSE_hist(1:iter_count);
    iter_time = iter_time(1:iter_count);
    cum_time = cum_time(1:iter_count);
    final_err = param_err(end);
    total_time = cum_time(end);
else
    theta_hist = zeros(n, 0);
    param_err = zeros(0, 1);
    rel_change = zeros(0, 1);
    RMSE_hist = zeros(0, 1);
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
    'iter_time', iter_time, ...
    'cum_time', cum_time, ...
    'final_err', final_err, ...
    'total_time', total_time, ...
    'iterations', iter_count);
end

function [theta_new, aux_state] = method_parameter_update(method_name, method_cfg, Phi_hat, c, theta_hat, iter_idx)
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
        theta_new = ws_rgls_update(Phi_hat, c, method_cfg);

    case 'WS-GNI'
        theta_new = ws_gni_update(Phi_hat, c, theta_hat, method_cfg);

    case {'WS_GGHAM_1_DH', 'WS-GGHAM-1-DH'}
        theta_new = ws_ggham_1_dh_update(Phi_hat, c, theta_hat, method_cfg);

    case 'WS-GGHAM-2-I'
        theta_new = ws_ggham_2_i_update(Phi_hat, c, theta_hat, method_cfg);

    case {'WS_GGHAM_2_DH', 'WS-GGHAM-2-DH'}
        theta_new = ws_ggham_2_dh_update(Phi_hat, c, theta_hat, method_cfg);

    case {'WS-LGHAM-1', 'WS_LGHAM_1', 'WS-LGHAM-1-PINV', 'WS_LGHAM_1_PINV', 'WS-LGHAM-1-TIK', 'WS_LGHAM_1_TIK'}
        theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, get_method_order(method_cfg, 1));

    case {'WS-LGHAM-2', 'WS_LGHAM_2', 'WS-LGHAM-2-TIK', 'WS_LGHAM_2_TIK', 'WS-LGHAM-2-PINV', 'WS_LGHAM_2_PINV'}
        theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, get_method_order(method_cfg, 2));

    case {'WS-LGHAM-3', 'WS_LGHAM_3', 'WS-LGHAM-3-TIK', 'WS_LGHAM_3_TIK', 'WS-LGHAM-3-PINV', 'WS_LGHAM_3_PINV'}
        theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, get_method_order(method_cfg, 3));

    case {'WS-GGHAM-1', 'WS-GGHAM-2'}
        aux_state.status = 'not_implemented';
        aux_state.message = sprintf('%s placeholder selected. Add its theta-update rule in method_parameter_update().', method_name);

    otherwise
        aux_state.status = 'not_implemented';
        aux_state.message = sprintf('Unknown method "%s". Add its config and update rule before running it.', method_name);
end
end

function theta_new = ws_rgls_update(Phi_hat, c, method_cfg)
theta_ls = solve_stable_least_squares(Phi_hat, c, method_cfg);
theta_new = project_stable_theta(theta_ls, method_cfg); % can be deactivated in config file
end

function theta_new = ws_ggham_2_i_update(Phi_hat, c, theta_hat, method_cfg)
eta = 1.0;
if isfield(method_cfg, 'eta')
    eta = method_cfg.eta;
end

residual = c - Phi_hat * theta_hat;
gradient_root = Phi_hat.' * residual;
H = Phi_hat.' * Phi_hat;

theta_1 = eta * gradient_root;
theta_new = theta_hat + 2 * theta_1 - eta * (H * theta_1);
end

function theta_new = ws_ggham_2_dh_update(Phi_hat, c, theta_hat, method_cfg)
eta = 1.0;
if isfield(method_cfg, 'eta')
    eta = method_cfg.eta;
end

lambda_reg = 0;
if isfield(method_cfg, 'hessian_regularization')
    lambda_reg = method_cfg.hessian_regularization;
end

residual = c - Phi_hat * theta_hat;
gradient_root = Phi_hat.' * residual;
H = Phi_hat.' * Phi_hat;

% Two-step implementation of eq. (26):
% theta_1 = eta * diag(H)^(-1) * gradient_root
% theta_{k+1} = theta_k + 2*theta_1 - eta * diag(H)^(-1) * (H * theta_1)
diag_H = diag(H);
if lambda_reg > 0
    diag_H = diag_H + lambda_reg;
end

theta_1 = eta * (gradient_root ./ diag_H);
theta_new = theta_hat + 2 * theta_1 - eta * ((H * theta_1) ./ diag_H);
end

function theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, order)
if order < 1 || order > 3
    error('WS-LGHAM is currently implemented only for orders 1 to 3.');
end

eps0 = get_eps0(method_cfg);
eta = method_cfg.eta;

residual = c - Phi_hat * theta_hat;
J_theta = 0.5 * (residual.' * residual);
gradient_root = Phi_hat.' * residual;
H = Phi_hat.' * Phi_hat;

theta_1 = solve_loss_root_inverse(gradient_root, J_theta - eps0, method_cfg);
theta_new = theta_hat + eta*theta_1;

if order >= 2
    theta_2 = theta_1 + solve_loss_root_inverse(gradient_root, gradient_root.' * theta_1, method_cfg);
    theta_new = theta_new + eta*theta_2;
end

if order >= 3
    rhs_3 = 0.5 * (theta_1.' * H * theta_1) + gradient_root.' * theta_2;
    theta_3 = theta_2 + solve_loss_root_inverse(gradient_root, rhs_3, method_cfg);
    theta_new = theta_new + eta*theta_3;
end
end

function eps0 = get_eps0(method_cfg)
eps0 = 0;
if isfield(method_cfg, 'eps_0')
    eps0 = method_cfg.eps_0;
elseif isfield(method_cfg, 'epsilon0')
    eps0 = method_cfg.epsilon0;
end
end

function order = get_method_order(method_cfg, default_order)
order = default_order;
if isfield(method_cfg, 'order')
    order = method_cfg.order;
end
end

function direction = solve_loss_root_inverse(gradient_root, rhs_scalar, method_cfg)

solver = 'pinv';
if isfield(method_cfg, 'inverse_solver')
    solver = lower(method_cfg.inverse_solver);
end

gg = gradient_root.' * gradient_root;

switch solver
    case 'pinv'
        if gg <= eps
            direction = zeros(size(g));
        else
            direction = gradient_root'\ rhs_scalar;
        end

    case 'tikhonov'
        lambda = 0;
        if isfield(method_cfg, 'tikhonov_lambda')
            lambda = method_cfg.tikhonov_lambda;
        end
        direction = lsqminnorm(gradient_root', rhs_scalar, RegularizationFactor=lambda);

    otherwise
        error('Unknown WS-LGHAM inverse_solver "%s".', solver);
end
end

function theta_new = ws_gni_update(Phi_hat, c, theta_hat, method_cfg)
step_size = 1.0;
if isfield(method_cfg, 'step_size')
    step_size = method_cfg.step_size;
end

lambda_reg = 0;
if isfield(method_cfg, 'hessian_regularization')
    lambda_reg = method_cfg.hessian_regularization;
end

residual = c - Phi_hat * theta_hat;
gradient = Phi_hat.' * residual;
H = Phi_hat.' * Phi_hat;

if lambda_reg > 0
    H = H + lambda_reg * eye(size(H));
end

if rcond(H) < 1e-12
    direction = pinv(H) * gradient;
else
    direction = H \ gradient;
end

theta_new = theta_hat + step_size * direction;
end

function theta_new = ws_ggham_1_dh_update(Phi_hat, c, theta_hat, method_cfg)
eta = 1.0;
if isfield(method_cfg, 'eta')
    eta = method_cfg.eta;
end

lambda_reg = 0;
if isfield(method_cfg, 'hessian_regularization')
    lambda_reg = method_cfg.hessian_regularization;
end

residual = c - Phi_hat * theta_hat;
gradient = Phi_hat.' * residual;

% Diagonal Hessian approximation: diag(Phi' * Phi)
diag_H = sum(Phi_hat.^2, 1).';
if lambda_reg > 0
    diag_H = diag_H + lambda_reg;
end

direction = gradient ./ (diag_H);

theta_new = theta_hat + eta * direction;
end

function init_state = initialize_method_state(method_cfg, r, c, na, nb, lambda_g, n)
alpha_hat = 1e-6 * ones(lambda_g, 1);
e_hat = 1e-6 * ones(lambda_g, 1);
theta_hat = 1e-6 * ones(n, 1);

if ~isfield(method_cfg, 'initialization')
    init_state = struct('alpha_hat', alpha_hat, 'e_hat', e_hat, 'theta_hat', theta_hat);
    return;
end

init_cfg = method_cfg.initialization;

if isfield(init_cfg, 'theta_init')
    theta_init = init_cfg.theta_init;
    if isscalar(theta_init)
        theta_hat = theta_init * ones(n, 1);
    else
        theta_hat = theta_init(:);
        assert(numel(theta_hat) == n, 'theta_init must have length n.');
    end
end

if isfield(init_cfg, 'mode') && strcmpi(init_cfg.mode, 'physical')
    a0 = zeros(na, 1);
    b0 = zeros(nb, 1);
    if nb >= 1
        b0(1) = 1;
    end

    if isfield(init_cfg, 'linear_init_a')
        a0 = fit_init_vector(init_cfg.linear_init_a, na, 0);
    end
    if isfield(init_cfg, 'linear_init_b')
        b0 = fit_init_vector(init_cfg.linear_init_b, nb, 0);
        if nb >= 1 && ~any(abs(b0) > 0)
            b0(1) = 1;
        end
    end

    den_lin_0 = [1; a0];
    num_lin_0 = [0; b0];
    alpha_hat = filter(num_lin_0.', den_lin_0.', r);
    alpha_hat = alpha_hat(:);

    e_mode = 'output_residual';
    if isfield(init_cfg, 'e_init_mode')
        e_mode = init_cfg.e_init_mode;
    end

    switch lower(e_mode)
        case 'output_residual'
            e_hat = c - alpha_hat;
        case 'zeros'
            e_hat = zeros(lambda_g, 1);
        otherwise
            error('Unknown e_init_mode "%s".', e_mode);
    end
else
    if isfield(init_cfg, 'alpha_init')
        alpha_init = init_cfg.alpha_init;
        if isscalar(alpha_init)
            alpha_hat = alpha_init * ones(lambda_g, 1);
        else
            alpha_hat = alpha_init(:);
            assert(numel(alpha_hat) == lambda_g, 'alpha_init must have length lambda_g.');
        end
    end

    if isfield(init_cfg, 'e_init')
        e_init = init_cfg.e_init;
        if isscalar(e_init)
            e_hat = e_init * ones(lambda_g, 1);
        else
            e_hat = e_init(:);
            assert(numel(e_hat) == lambda_g, 'e_init must have length lambda_g.');
        end
    end
end

init_state = struct('alpha_hat', alpha_hat, 'e_hat', e_hat, 'theta_hat', theta_hat);
end

function vec_out = fit_init_vector(vec_in, target_len, fill_value)
if nargin < 3
    fill_value = 0;
end

if isscalar(vec_in)
    vec_out = vec_in * ones(target_len, 1);
    return;
end

vec_in = vec_in(:);
vec_out = fill_value * ones(target_len, 1);
n_copy = min(numel(vec_in), target_len);
vec_out(1:n_copy) = vec_in(1:n_copy);
end

function theta_ls = solve_stable_least_squares(Phi_hat, c, method_cfg)
lambda_reg = 0;

if isfield(method_cfg, 'regularization')
    reg_cfg = method_cfg.regularization;
    if isfield(reg_cfg, 'enabled') && reg_cfg.enabled
        if isfield(reg_cfg, 'lambda')
            lambda_reg = reg_cfg.lambda;
        else
            lambda_reg = 0;
        end
    end
end

if lambda_reg > 0
    ncols = size(Phi_hat, 2);
    theta_ls = (Phi_hat.' * Phi_hat + lambda_reg * eye(ncols)) \ (Phi_hat.' * c);
else
    theta_ls = Phi_hat \ c;
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
    iter_axis = 0:(results(idx).iterations - 1);
    y_param_err = max(results(idx).param_err, eps);
    plot(iter_axis, y_param_err, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
set(gca, 'YScale', 'log');
grid on; box on; xlabel('Iteration'); ylabel('err (%)');
title('Parameter Error vs Iteration'); legend('Location', 'best');

subplot(2, 2, 2); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    y_param_err = max(results(idx).param_err, eps);
    plot(results(idx).cum_time, y_param_err, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
set(gca, 'YScale', 'log');
grid on; box on; xlabel('Compute time (s)'); ylabel('err (%)');
title('Parameter Error vs Compute Time'); legend('Location', 'best');

subplot(2, 2, 3); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    iter_axis = 0:(results(idx).iterations - 1);
    y_rmse = max(results(idx).RMSE_hist, eps);
    plot(iter_axis, y_rmse, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
set(gca, 'YScale', 'log');
grid on; box on; xlabel('Iteration'); ylabel('RMSE');
title('RMSE vs Iteration'); legend('Location', 'best');

subplot(2, 2, 4); hold on;
for j = 1:numel(valid_idx)
    idx = valid_idx(j);
    y_rmse = max(results(idx).RMSE_hist, eps);
    plot(results(idx).cum_time, y_rmse, 'LineWidth', 1.6, ...
        'Color', colors(j, :), 'DisplayName', results(idx).name);
end
set(gca, 'YScale', 'log');
grid on; box on; xlabel('Compute time (s)'); ylabel('RMSE');
title('RMSE vs Compute Time'); legend('Location', 'best');

sgtitle('Wiener System Identification - Method Comparison', 'FontSize', 13, 'FontWeight', 'bold');
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
        iter_axis = 0:(results(idx).iterations - 1);
        plot(iter_axis, results(idx).theta_hist(i, :), ...
            'LineWidth', 1.4, 'Color', colors(j, :), 'DisplayName', results(idx).name);
    end
    yline(theta_true(i), 'k--', 'LineWidth', 1.1, 'DisplayName', 'True');
    grid on; box on;
    xlabel('Iteration');
    ylabel(['$\hat{', param_labels{i}, '}$'], 'Interpreter', 'latex');
    title(['Parameter ', param_labels{i}]);
    legend('Location', 'best');
end

sgtitle('Parameter Trajectories Across Methods', 'FontSize', 12, 'FontWeight', 'bold');
end
