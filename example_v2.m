%% WS-GGI Algorithm - Wiener Nonlinear System with AR Noise (Example 1)
%
% Reference: Lv et al., "Parameter estimation of Wiener nonlinear systems
% based on gradient iteration theory and zebra optimization algorithm",
% Journal of the Franklin Institute 363 (2026) 108499.
%
% System structure (Fig. 1 of the paper):
%   r(t) --> G(z) --> alpha(t) --> F(.) --> beta(t) --[+]--> c(t)
%                                                        ^
%                                        nu(t) --> w(z) --> e(t)
%
% Example 1 true parameters:
%   alpha(t) = (0.23*z^-1 + 0.98*z^-2) / (1 - 0.31*z^-1 - 0.27*z^-2) * r(t)
%   beta(t)  = alpha(t) + 0.32*alpha(t)^2
%   e(t)     = 1/(1 - 0.40*z^-1) * nu(t)
%   theta    = [a1,a2,b1,b2,f2,d1] = [-0.31,-0.27,0.23,0.98,0.32,-0.40]

clear; clc; close all;
rng(42);

%% 1. True system parameters
na = 2;
nb = 2;
nf = 2;
nd = 1;
n  = na + nb + (nf - 1) + nd;

a_true = [-0.31; -0.27];
b_true = [ 0.23;  0.98];
f_true = 0.32;
d_true = -0.40;

theta_true = [a_true; b_true; f_true; d_true];

%% 2. Algorithm hyper-parameters
lambda_g = 1000;
K_max = 240;
conv_threshold = 1e-8;
sigma_nu = 0.10;

%% 3. Generate input / noise / true output
burn_in = 100;
N_total = lambda_g + burn_in;
r_full = randn(N_total, 1);
nu_full = sigma_nu * randn(N_total, 1);

poly_coeffs = [flipud(f_true(:)); 1; 0];

den_lin = [1; a_true(:)];
num_lin = [0; b_true(:)];
alpha_full = filter(num_lin.', den_lin.', r_full);

den_ar = [1; -d_true(:)];
e_full = filter(1, den_ar.', nu_full);

beta_full = polyval(poly_coeffs.', alpha_full);
c_full = beta_full + e_full;

keep_idx = (burn_in + 1):(burn_in + lambda_g);
r = r_full(keep_idx);
nu = nu_full(keep_idx);
c = c_full(keep_idx);

%% 4. WS-GGI Algorithm
alpha_hat = 1e-6 * ones(lambda_g, 1);
e_hat = 1e-6 * ones(lambda_g, 1);
nu_hat = 1e-6 * ones(lambda_g, 1);
theta_hat = 1e-6 * ones(n, 1);

param_err = zeros(K_max, 1);
rel_change = zeros(K_max, 1);
theta_hist = zeros(n, K_max);
RMSE_hist = zeros(K_max, 1);
MAE_hist = zeros(K_max, 1);

param_names = build_param_names(na, nb, nf, nd);
fprintf('Iter    ');
fprintf('%-10s  ', param_names{:});
fprintf('%-10s\n', 'err(%)');
fprintf('%s\n', repmat('-', 1, 12 * (n + 2)));

for k = 1:K_max

    %% (a) Build state matrix Phi_hat^(k)  [lambda_g x n]
    % Regressor structure:
    %   phi_hat(t) = [-alpha(t-1..t-na), r(t-1..t-nb), alpha(t).^2..alpha(t).^nf, -e(t-1..t-nd)]
    %
    % Unavailable lagged samples are zero-padded. This keeps Phi_hat valid
    % after burn-in removal because no access is made outside the retained
    % signal bounds.
    Phi_hat = build_state_matrix(alpha_hat, e_hat, r, na, nb, nf, nd);

    %% (b) Iteration step size  Eq.(30)
    delta = 1.5 / max(norm(Phi_hat, 'fro')^2, eps);

    %% (c) Update parameter vector  Eq.(28)
    theta_new = theta_hat + delta * (Phi_hat' * (c - Phi_hat * theta_hat));

    %% Extract parameter estimates from theta_new
    a_hat_v = theta_new(1:na);
    b_hat_v = theta_new(na + 1:na + nb);
    f_hat_v = theta_new(na + nb + 1:na + nb + (nf - 1));
    d_hat_v = theta_new(na + nb + (nf - 1) + 1:end);

    %% (d) Update nu_hat^(k)  Eq.(24)
    nu_hat = c - Phi_hat * theta_new;

    %% (e) Update e_hat^(k)
    e_hat_new = zeros(lambda_g, 1);
    for t = 1:lambda_g
        acc = nu_hat(t);
        for i = 1:nd
            if t > i
                acc = acc - d_hat_v(i) * e_hat_new(t - i);
            end
        end
        e_hat_new(t) = acc;
    end
    e_hat = e_hat_new;

    %% (f) Update alpha_hat^(k)
    alpha_hat_new = zeros(lambda_g, 1);
    for t = 1:lambda_g
        acc = 0;
        for i = 1:na
            if t > i
                acc = acc - a_hat_v(i) * alpha_hat_new(t - i);
            end
        end
        for i = 1:nb
            if t > i
                acc = acc + b_hat_v(i) * r(t - i);
            end
        end
        alpha_hat_new(t) = acc;
    end
    alpha_hat = alpha_hat_new;

    %% Convergence & metrics
    rel_change(k) = norm(theta_new - theta_hat) / (norm(theta_hat) + 1e-15);
    theta_hat = theta_new;
    theta_hist(:, k) = theta_hat;

    param_err(k) = norm(theta_hat - theta_true) / norm(theta_true) * 100;

    c_sim = simulate_wiener(r, nu, theta_hat, na, nb, nf, nd, lambda_g);
    res = c - c_sim;
    RMSE_hist(k) = sqrt(mean(res.^2));
    MAE_hist(k) = mean(abs(res));

    if ismember(k, [5 10 15 30 50 70 100 240]) || k == K_max
        fprintf('%-6d  ', k);
        fprintf('%-10.5f  ', theta_hat);
        fprintf('%-10.5f\n', param_err(k));
    end

    if k > 1 && rel_change(k) < conv_threshold
        fprintf('\n*** Converged at iteration %d ***\n', k);
        K_max = k;
        break;
    end
end

%% 5. Final results table
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('FINAL PARAMETER ESTIMATES vs. TRUE VALUES\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('  %-8s  %-12s  %-12s\n', 'Param', 'Estimated', 'True');
fprintf('%s\n', repmat('-', 1, 40));
for i = 1:n
    fprintf('  %-8s  %-12.5f  %-12.5f\n', param_names{i}, theta_hat(i), theta_true(i));
end
fprintf('%s\n', repmat('-', 1, 40));
fprintf('  Final err(%%) = %.5f\n', param_err(K_max));

%% 6. Plots
c_sim_final = simulate_wiener(r, nu, theta_hat, na, nb, nf, nd, lambda_g);

figure('Name', 'WS-GGI Results', 'Color', 'w', 'Position', [100 80 1000 750]);

subplot(2, 2, [1 2]);
plot(1:lambda_g, c, 'b-', 'LineWidth', 0.8, 'DisplayName', 'Observed Output'); hold on;
plot(1:lambda_g, c_sim_final, 'r.', 'MarkerSize', 4, 'DisplayName', 'Simulated Output / WS-GGI');
legend('Location', 'best'); grid on;
xlabel('Time'); ylabel('c(t)');
title('Observation and Simulation Output of WS-GGI');
xlim([1 lambda_g]);

subplot(2, 2, 3);
semilogy(1:K_max, param_err(1:K_max), 'b-', 'LineWidth', 1.8);
grid on; xlabel('Iteration'); ylabel('err (%)');
title('Parameter Error Convergence');
hold on;
marks = [5 10 15 30 50 70 100 240];
marks = marks(marks <= K_max);
plot(marks, param_err(marks), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 5);

subplot(2, 2, 4);
plot(1:K_max, RMSE_hist(1:K_max), 'b-', 'LineWidth', 1.5, 'DisplayName', 'RMSE'); hold on;
plot(1:K_max, MAE_hist(1:K_max), 'r--', 'LineWidth', 1.5, 'DisplayName', 'MAE');
grid on; legend; xlabel('Iteration'); ylabel('Error');
title('RMSE and MAE vs Iteration');

sgtitle('WS-GGI: Wiener System with AR Noise - Example 1', 'FontSize', 13, 'FontWeight', 'bold');

%% 7. Parameter trajectory plot
figure('Name', 'Parameter Trajectories', 'Color', 'w', 'Position', [150 100 900 550]);
param_labels = build_plot_labels(na, nb, nf, nd);
colors = lines(n);
for i = 1:n
    subplot(ceil(n / 3), 3, i);
    plot(1:K_max, theta_hist(i, 1:K_max), 'Color', colors(i, :), 'LineWidth', 1.5); hold on;
    yline(theta_true(i), 'k--', 'LineWidth', 1.2);
    grid on;
    xlabel('Iteration');
    ylabel(['$\hat{', param_labels{i}, '}$'], 'Interpreter', 'latex');
    title(['Parameter ', param_labels{i}]);
    legend('Estimate', 'True', 'Location', 'best');
end
sgtitle('Parameter Convergence Trajectories - WS-GGI', 'FontSize', 12, 'FontWeight', 'bold');

fprintf('\nDone. All figures generated.\n');


%% Local functions
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

den_ar = [1; -d_v(:)];
e = filter(1, den_ar.', nu);

c_sim = beta(:) + e(:);

if nargin >= 8
    c_sim = c_sim(1:T);
end
end
