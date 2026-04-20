%% WS-GGI Algorithm — Wiener Nonlinear System with AR Noise (Example 1)
%
% Reference: Lv et al., "Parameter estimation of Wiener nonlinear systems
%   based on gradient iteration theory and zebra optimization algorithm",
%   Journal of the Franklin Institute 363 (2026) 108499.
%
% System structure (Fig. 1 of the paper):
%   r(t) --> G(sigma) --> alpha(t) --> F(·) --> beta(t) --[+]--> c(t)
%                                                          ^
%                                          nu(t) --> w(sigma) --> e(t)
%
% Example 1 true parameters:
%   alpha(t) = (0.23*z^-1 + 0.98*z^-2) / (1 - 0.31*z^-1 - 0.27*z^-2) * r(t)
%   beta(t)  = alpha(t) + 0.32*alpha(t)^2
%   e(t)     = 1/(1 - 0.40*z^-1) * nu(t)
%   theta    = [a1,a2, b1,b2, f2, d1] = [-0.31,-0.27, 0.23,0.98, 0.32,-0.40]

clear; clc; close all;
rng(42);  % reproducibility

%% ── 1. True system parameters ─────────────────────────────────────────────
na = 2; nb = 2; nf = 2; nd = 1;
n  = na + nb + (nf-1) + nd;   % dimension = 6  (f1=1 is known, not estimated)

a_true = [-0.31; -0.27];   % denominator of G(sigma)
b_true = [ 0.23;  0.98];   % numerator   of G(sigma)
f2_true =  0.32;            % nonlinear coefficient (f1=1 fixed)
d_true  = [-0.40];          % AR noise denominator

theta_true = [a_true; b_true; f2_true; d_true];   % 6-vector

%% ── 2. Algorithm hyper-parameters ─────────────────────────────────────────
lambda_g = 1000;   % data length  (paper uses 1000)
K_max    = 240;    % maximum iterations
zeta     = 1e-8;   % convergence threshold on relative change
sigma_nu = 0.10;   % noise standard deviation  (variance = 0.1^2)

%% ── 3. Generate input / noise / true output ────────────────────────────────
N_total  = lambda_g + 100;          % extra samples for transient burn-in
r_full   = randn(N_total, 1);       % persistent excitation (zero-mean, unit-var)
nu_full  = sigma_nu * randn(N_total, 1);

alpha_full = zeros(N_total, 1);
e_full     = zeros(N_total, 1);
c_full     = zeros(N_total, 1);

for t = 3:N_total
    % Linear subsystem  Eq.(10): alpha(t) = -sum a_i*alpha(t-i) + sum b_i*r(t-i)
    alpha_full(t) = -a_true(1)*alpha_full(t-1) - a_true(2)*alpha_full(t-2) ...
                   +  b_true(1)*r_full(t-1)    +  b_true(2)*r_full(t-2);

    % AR noise  Eq.(11): e(t) = -d1*e(t-1) + nu(t)
    e_full(t) = -d_true(1)*e_full(t-1) + nu_full(t);

    % Output  Eq.(9):  c(t) = beta(t) + e(t)
    beta_t      = alpha_full(t) + f2_true * alpha_full(t)^2;  % Eq.(8), f1=1
    c_full(t)   = beta_t + e_full(t);
end

% Discard burn-in, keep exactly lambda_g samples
r  = r_full(101:100+lambda_g);
nu = nu_full(101:100+lambda_g);
c  = c_full(101:100+lambda_g);    % observation vector C(lambda_g)

%% ── 4. WS-GGI Algorithm ────────────────────────────────────────────────────
%
%  Each outer iteration k:
%   (a) Build Phi_hat^(k) from alpha_hat^(k-1) and e_hat^(k-1)   Eq.(23),(27)
%   (b) delta^(k) = 1.5 / ||Phi_hat^(k)||_F^2                    Eq.(30)
%   (c) theta_hat^(k) = theta_hat^(k-1)
%                     + delta^(k) * Phi_hat^(k)' * [C - Phi_hat^(k)*theta_hat^(k-1)]   Eq.(28)
%   (d) nu_hat^(k)(t) = c(t) - phi_hat^(k)(t)' * theta_hat^(k)   Eq.(24)
%   (e) e_hat^(k)(t)  = -d1_hat*e_hat^(k)(t-1) + nu_hat^(k)(t)   Eq.(25)
%   (f) alpha_hat^(k)(t) = -sum a_hat*alpha_hat^(k)(t-i)
%                        + sum b_hat*r(t-i)                       Eq.(26)

% ── Initialise ────────────────────────────────────────────────────────────
alpha_hat = 1e-6 * ones(lambda_g, 1);
e_hat     = 1e-6 * ones(lambda_g, 1);
nu_hat    = 1e-6 * ones(lambda_g, 1);
theta_hat = 1e-6 * ones(n, 1);

% Storage
param_err  = zeros(K_max, 1);   % err(%) = ||theta_hat - theta_true|| / ||theta_true|| * 100
rel_change = zeros(K_max, 1);   % convergence check
theta_hist = zeros(n, K_max);
RMSE_hist  = zeros(K_max, 1);
MAE_hist   = zeros(K_max, 1);

fprintf('%-6s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n', ...
        'Iter','a1','a2','b1','b2','f2','d1','err(%)');
fprintf('%s\n', repmat('-',1,80));

for k = 1:K_max

    %% (a) Build state matrix Phi_hat^(k)  [lambda_g x n]
    %      phi_hat^(k)(t) = [-alpha^(k-1)(t-1), -alpha^(k-1)(t-2),
    %                         r(t-1), r(t-2),
    %                         (alpha^(k-1)(t))^2,
    %                        -e^(k-1)(t-1)]
    Phi_hat = zeros(lambda_g, n);

    for t = 1:lambda_g
        alpha_tm1 = (t > 1) * alpha_hat(t-1); % ERROR: it tries to save samples from before the signal starts!
        alpha_tm2 = (t > 2) * alpha_hat(t-2);
        r_tm1     = (t > 1) * r(t-1);
        r_tm2     = (t > 2) * r(t-2);
        alpha_t_sq = alpha_hat(t)^2;
        e_tm1     = (t > 1) * e_hat(t-1);

        Phi_hat(t, :) = [-alpha_tm1, -alpha_tm2, ...
                          r_tm1,      r_tm2,      ...
                          alpha_t_sq,             ...
                         -e_tm1];
    end

    %% (b) Iteration step size  Eq.(30)
    delta = 1.5 / norm(Phi_hat, 'fro')^2;

    %% (c) Update parameter vector  Eq.(28)
    theta_new = theta_hat + delta * (Phi_hat' * (c - Phi_hat * theta_hat));

    %% Extract parameter estimates from theta_new
    a_hat_v  = theta_new(1:na);               % [a1; a2]
    b_hat_v  = theta_new(na+1 : na+nb);       % [b1; b2]
    f2_hat_v = theta_new(na+nb+1);            % f2
    d_hat_v  = theta_new(na+nb+2 : end);      % [d1]

    %% (d) Update nu_hat^(k)  Eq.(24)
    nu_hat = c - Phi_hat * theta_new;

    %% (e) Update e_hat^(k)  Eq.(25):  e(t) = -d1*e(t-1) + nu(t)
    e_hat_new = zeros(lambda_g, 1);
    for t = 1:lambda_g
        acc = nu_hat(t);
        for i = 1:nd
            if t > i
                acc = acc - d_hat_v(i) * e_hat_new(t-i);
            end
        end
        e_hat_new(t) = acc;
    end
    e_hat = e_hat_new;

    %% (f) Update alpha_hat^(k)  Eq.(26)
    alpha_hat_new = zeros(lambda_g, 1);
    for t = 1:lambda_g
        acc = 0;
        for i = 1:na
            if t > i; acc = acc - a_hat_v(i) * alpha_hat_new(t-i); end
        end
        for i = 1:nb
            if t > i; acc = acc + b_hat_v(i) * r(t-i); end
        end
        alpha_hat_new(t) = acc;
    end
    alpha_hat = alpha_hat_new;

    %% Convergence & metrics
    rel_change(k) = norm(theta_new - theta_hat) / (norm(theta_hat) + 1e-15);
    theta_hat     = theta_new;
    theta_hist(:,k) = theta_hat;

    param_err(k) = norm(theta_hat - theta_true) / norm(theta_true) * 100;

    % Simulate output with current parameters for RMSE/MAE
    c_sim = simulate_wiener(r, nu, theta_hat, na, nb, nd, lambda_g);
    res   = c - c_sim;
    RMSE_hist(k) = sqrt(mean(res.^2));
    MAE_hist(k)  = mean(abs(res));

    % Print at selected iterations (matching Table 2 in paper)
    if ismember(k, [5 10 15 30 50 70 100 240]) || k == K_max
        fprintf('%-6d  %-10.5f  %-10.5f  %-10.5f  %-10.5f  %-10.5f  %-10.5f  %-10.5f\n', ...
                k, theta_hat(1), theta_hat(2), theta_hat(3), ...
                   theta_hat(4), theta_hat(5), theta_hat(6), param_err(k));
    end

    % Stop early if converged
    if k > 1 && rel_change(k) < zeta
        fprintf('\n  *** Converged at iteration %d ***\n', k);
        K_max = k; break;
    end
end

%% ── 5. Final results table ─────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,60));
fprintf('  FINAL PARAMETER ESTIMATES vs. TRUE VALUES\n');
fprintf('%s\n', repmat('=',1,60));
names = {'a1','a2','b1','b2','f2','d1'};
fprintf('  %-8s  %-12s  %-12s\n','Param','Estimated','True');
fprintf('%s\n', repmat('-',1,40));
for i = 1:n
    fprintf('  %-8s  %-12.5f  %-12.5f\n', names{i}, theta_hat(i), theta_true(i));
end
fprintf('%s\n', repmat('-',1,40));
fprintf('  Final err(%%) = %.5f\n', param_err(K_max));

%% ── 6. Plots ───────────────────────────────────────────────────────────────
c_sim_final = simulate_wiener(r, nu, theta_hat, na, nb, nd, lambda_g);

figure('Name','WS-GGI Results','Color','w','Position',[100 80 1000 750]);

% Plot 1: Observed vs Simulated Output
subplot(2,2,[1 2]);
plot(1:lambda_g, c, 'b-', 'LineWidth', 0.8, 'DisplayName','Observed Output'); hold on;
plot(1:lambda_g, c_sim_final, 'r.', 'MarkerSize', 4, 'DisplayName','Simulated Output / WS-GGI');
legend('Location','best'); grid on;
xlabel('Time'); ylabel('c(t)');
title('Observation and Simulation Output of WS-GGI');
xlim([1 lambda_g]);

% Plot 2: Parameter error convergence
subplot(2,2,3);
semilogy(1:K_max, param_err(1:K_max), 'b-', 'LineWidth', 1.8);
grid on; xlabel('Iteration'); ylabel('err (%)');
title('Parameter Error Convergence');
hold on;
marks = [5 10 15 30 50 70 100 240];
marks = marks(marks <= K_max);
plot(marks, param_err(marks), 'ro', 'MarkerFaceColor','r', 'MarkerSize', 5);

% Plot 3: RMSE / MAE
subplot(2,2,4);
plot(1:K_max, RMSE_hist(1:K_max), 'b-', 'LineWidth',1.5,'DisplayName','RMSE'); hold on;
plot(1:K_max, MAE_hist(1:K_max),  'r--','LineWidth',1.5,'DisplayName','MAE');
grid on; legend; xlabel('Iteration'); ylabel('Error');
title('RMSE and MAE vs Iteration');

sgtitle('WS-GGI: Wiener System with AR Noise — Example 1', 'FontSize',13,'FontWeight','bold');

%% ── 7. Parameter trajectory plot ──────────────────────────────────────────
figure('Name','Parameter Trajectories','Color','w','Position',[150 100 900 550]);
param_labels = {'a_1','a_2','b_1','b_2','f_2','d_1'};
colors = lines(n);
for i = 1:n
    subplot(2,3,i);
    plot(1:K_max, theta_hist(i,1:K_max), 'Color', colors(i,:), 'LineWidth', 1.5); hold on;
    yline(theta_true(i), 'k--', 'LineWidth', 1.2);
    grid on;
    xlabel('Iteration'); ylabel(['$\hat{',param_labels{i},'}$'], 'Interpreter','latex');
    title(['Parameter ',param_labels{i}]);
    legend('Estimate','True','Location','best');
end
sgtitle('Parameter Convergence Trajectories — WS-GGI','FontSize',12,'FontWeight','bold');

fprintf('\nDone. All figures generated.\n');


%% ══════════════════════════════════════════════════════════════════════════
%%  LOCAL FUNCTION: simulate Wiener output given parameter vector
%% ══════════════════════════════════════════════════════════════════════════
function c_sim = simulate_wiener(r, nu, theta, na, nb, nd, T)
    a_v  = theta(1:na);
    b_v  = theta(na+1:na+nb);
    f2_v = theta(na+nb+1);
    d_v  = theta(na+nb+2:end);

    alpha = zeros(T,1);
    e_s   = zeros(T,1);
    c_sim = zeros(T,1);

    for t = 1:T
        % linear subsystem
        val = 0;
        for i = 1:na
            if t > i; val = val - a_v(i)*alpha(t-i); end
        end
        for i = 1:nb
            if t > i; val = val + b_v(i)*r(t-i); end
        end
        alpha(t) = val;

        % AR noise
        en = nu(t);
        for i = 1:nd
            if t > i; en = en - d_v(i)*e_s(t-i); end
        end
        e_s(t) = en;

        % output: beta + e,  f1 = 1 (fixed)
        c_sim(t) = alpha(t) + f2_v*alpha(t)^2 + e_s(t);
    end
end