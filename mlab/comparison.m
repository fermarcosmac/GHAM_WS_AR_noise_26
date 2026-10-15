function payload = comparison(experiment_cfg, options)
% Optional config/options allow small runs without changing the published settings.
%% Wiener System Identification - Method Comparison Harness
%
% This script compares multiple identification methods on the Wiener system
% with AR noise from Lv et al. (2026). Hyperparameters are loaded from
% optim_configs.json so new methods can be added without changing the main
% experiment loop.

if nargin < 2, options = struct(); end
script_dir = fileparts(mfilename('fullpath'));

%% 0. Experiment configuration
experiment_name = 'example_CSTR';
repo_root = fullfile(script_dir, '..');
if nargin < 1 || isempty(experiment_cfg)
    experiment_cfg = load_experiment_config(repo_root, experiment_name);
end
experiment_name = get_optional_field(options, 'experiment_name', experiment_name);
mode = upper(char(get_optional_field(experiment_cfg, 'mode', 'EXAMPLE')));
seed = get_optional_field(experiment_cfg, 'seed', 42);
N_MC = get_optional_field(experiment_cfg, 'N_MC', 1);
config_file = fullfile(script_dir, get_optional_field(experiment_cfg, 'matlab_optim_config', 'optim_configs.json'));
make_local_plots = get_optional_field(experiment_cfg, 'make_local_plots', strcmp(mode, 'EXAMPLE'));
results_dir = fullfile(repo_root, 'results', experiment_name);
results_file = fullfile(results_dir, sprintf('comparison_pl_%s.mat', lower(mode)));
results_file = get_optional_field(options, 'output_file', results_file);
make_local_plots = get_optional_field(options, 'make_local_plots', make_local_plots);
selected_methods = cellstr_value(experiment_cfg.selected_methods);

%% 1. True system parameters
system_cfg = experiment_cfg.system;
na = system_cfg.na;
nb = system_cfg.nb;
nf = system_cfg.nf;
nd = get_optional_field(system_cfg, 'nd', 0);
n = na + nb + (nf - 1) + nd;
theta_true = system_cfg.theta_true(:);
assert(numel(theta_true) == n, 'theta_true has %d entries, but dimensions require %d.', numel(theta_true), n);
param_names = build_param_names(na, nb, nf, nd);
param_labels = build_plot_labels(na, nb, nf, nd);

%% 2. Shared experiment settings
settings = experiment_cfg.settings;
lambda_g = settings.lambda_g;
K_max = settings.K_max;
conv_threshold = settings.conv_threshold;
sigma_nu = settings.sigma_nu;
burn_in = settings.burn_in;
enforce_fixed_iterations = get_optional_field(settings, 'enforce_fixed_iterations', strcmp(mode, 'MONTECARLO'));

%% 3. Generate input / noise / true output
rng(seed);
[r, nu, c] = generate_experiment_data(theta_true, na, nb, nf, nd, lambda_g, burn_in, sigma_nu);

%% 4. Load optimization configs
method_configs = load_method_configs(config_file);
method_configs = get_optional_field(options, 'method_configs', method_configs);

%% 5. Run selected methods
results = run_method_set(selected_methods, method_configs, r, nu, c, theta_true, ...
    na, nb, nf, nd, lambda_g, K_max, conv_threshold, enforce_fixed_iterations, true, experiment_cfg, ...
    struct('data_seed', seed, 'mc_index', 1, 'experiment_name', experiment_name));

mc_summary = [];
mc_runs = {results};
if strcmp(mode, 'MONTECARLO')
    mc_runs = cell(N_MC, 1);
    mc_runs{1} = results;
    for mc = 2:N_MC
        run_seed = seed + mc - 1;
        fprintf('\nMonte Carlo run %d/%d (seed %d)\n', mc, N_MC, run_seed);
        rng(run_seed);
        [r_mc, nu_mc, c_mc] = generate_experiment_data(theta_true, na, nb, nf, nd, lambda_g, burn_in, sigma_nu);
        mc_runs{mc} = run_method_set(selected_methods, method_configs, r_mc, nu_mc, c_mc, theta_true, ...
            na, nb, nf, nd, lambda_g, K_max, conv_threshold, enforce_fixed_iterations, false, experiment_cfg, ...
            struct('data_seed', run_seed, 'mc_index', mc, 'experiment_name', experiment_name));
    end
    mc_summary = summarize_montecarlo(mc_runs, selected_methods, theta_true, K_max);
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
payload = save_comparison_results(results_file, results, selected_methods, ...
    na, nb, nf, nd, theta_true, param_names, param_labels, ...
    lambda_g, K_max, conv_threshold, sigma_nu, burn_in, enforce_fixed_iterations, ...
    experiment_name, mode, seed, N_MC, struct('r', r, 'nu', nu, 'c', c), mc_summary, mc_runs, experiment_cfg);

if make_local_plots
    plot_method_metrics(results);
    plot_parameter_trajectories(results, theta_true, param_labels);
    fprintf('\nDone. Results saved and local figures generated.\n');
else
    fprintf('\nDone. Results saved for MATLAB plotting.\n');
end


end

%% Local functions
function results = run_method_set(selected_methods, method_configs, r, nu, c, theta_true, na, nb, nf, nd, lambda_g, K_max, conv_threshold, enforce_fixed_iterations, verbose, experiment_cfg, metadata)
results = [];
best_error_so_far = Inf;
for m = 1:numel(selected_methods)
    name = selected_methods{m};
    cfg = struct();
    if has_method_config(method_configs, name), cfg = get_method_config(method_configs, name); end
    if isfield(experiment_cfg, 'initialization'), cfg.initialization = experiment_cfg.initialization; end
    cfg.initialization = get_optional_field(cfg, 'initialization', struct());
    cfg.initialization.policy = get_optional_field(cfg.initialization, 'policy', 'legacy');
    defaults = struct('stopping', struct('gradient_tolerance',1e-6,'loss_tolerance',1e-8), ...
        'diagnostics', struct('enabled',true,'save_matrices',false));
    for section = {'stopping', 'diagnostics'}
        key = section{1};
        merged = defaults.(key);
        for source = {get_optional_field(cfg,key,struct()), get_optional_field(experiment_cfg,key,struct())}
            fields = fieldnames(source{1});
            for j = 1:numel(fields), merged.(fields{j}) = source{1}.(fields{j}); end
        end
        cfg.(key) = merged;
    end
    cfg.na = na; cfg.nb = nb; cfg.nf = nf; cfg.nd = nd;
    cfg.run_settings = struct('K_max',K_max,'conv_threshold',conv_threshold, ...
        'enforce_fixed_iterations',enforce_fixed_iterations);
    if contains(name,'LGHAM')
        cfg.best_error_so_far = best_error_so_far;
        if ~isfinite(best_error_so_far), cfg.best_error_so_far = 0; end
        cfg.resolved_epsilon0 = get_eps0(cfg);
    end
    result = run_identification_method(name, cfg, r, nu, c, theta_true, ...
        na, nb, nf, nd, lambda_g, K_max, conv_threshold, enforce_fixed_iterations, best_error_so_far, verbose, metadata);
    best_error_so_far = result.best_error_so_far;
    if isempty(results), results = result; else, results(end+1) = result; end %#ok<AGROW>
end
end

function [r, nu, c] = generate_experiment_data(theta_true, na, nb, nf, nd, lambda_g, burn_in, sigma_nu)
N_total = lambda_g + burn_in;
r_full = randn(N_total, 1);
nu_full = sigma_nu * randn(N_total, 1);
c_full = simulate_wiener(r_full, nu_full, theta_true, na, nb, nf, nd, N_total);
keep_idx = (burn_in + 1):(burn_in + lambda_g);
r = r_full(keep_idx);
nu = nu_full(keep_idx);
c = c_full(keep_idx);
end

function summary = summarize_montecarlo(mc_runs, selected_methods, theta_true, K_max)
N_MC = numel(mc_runs);
n_methods = numel(selected_methods);
n_params = numel(theta_true);
param_err_runs = NaN(n_methods, N_MC, K_max);
rmse_runs = NaN(n_methods, N_MC, K_max);
final_param_errors = NaN(n_methods, N_MC, n_params);
final_param_estimates = NaN(n_methods, N_MC, n_params);
final_errors = NaN(n_methods, N_MC);
total_time_runs = NaN(n_methods, N_MC);
statuses = cell(n_methods,N_MC); run_ids = cell(n_methods,N_MC);

for mc = 1:N_MC
    run_results = mc_runs{mc};
    for m = 1:n_methods
        idx = find(strcmp({run_results.name}, selected_methods{m}), 1);
        if isempty(idx)
            continue;
        end
        result = run_results(idx);
        statuses{m,mc} = result.status; run_ids{m,mc} = result.run_metadata.run_id;
        total_time_runs(m,mc) = result.total_time;
        if ~ismember(result.status,{'converged','stalled','max_iterations'}), continue; end
        param_err_runs(m, mc, :) = pad_curve(result.param_err, K_max);
        rmse_runs(m, mc, :) = pad_curve(result.RMSE_hist, K_max);
        final_errors(m, mc) = result.final_err;
        total_time_runs(m, mc) = result.total_time;
        if ~isempty(result.theta_hat)
            theta_hat = result.theta_hat(:);
            final_param_estimates(m, mc, :) = theta_hat;
            final_param_errors(m, mc, :) = theta_hat - theta_true(:);
        end
    end
end

summary = struct( ...
    'statuses',{statuses},'run_ids',{run_ids}, ...
    'method_names', {selected_methods}, ...
    'param_err_runs', param_err_runs, ...
    'param_err_mean', reshape(mean(param_err_runs,2,'omitnan'),n_methods,K_max), ...
    'param_err_stderr', reshape(std(param_err_runs,0,2,'omitnan')./sqrt(max(sum(isfinite(param_err_runs),2),1)),n_methods,K_max), ...
    'rmse_runs', rmse_runs, ...
    'rmse_mean', reshape(mean(rmse_runs,2,'omitnan'),n_methods,K_max), ...
    'rmse_stderr', reshape(std(rmse_runs,0,2,'omitnan')./sqrt(max(sum(isfinite(rmse_runs),2),1)),n_methods,K_max), ...
    'final_param_errors', final_param_errors, ...
    'final_param_estimates', final_param_estimates, ...
    'final_errors', final_errors, ...
    'total_time_runs', total_time_runs);
summary.param_err_count = reshape(sum(isfinite(param_err_runs),2),n_methods,K_max);
summary.rmse_count = reshape(sum(isfinite(rmse_runs),2),n_methods,K_max);
summary.param_err_stderr(summary.param_err_count<2) = NaN;
summary.rmse_stderr(summary.rmse_count<2) = NaN;
end

function y = pad_curve(x, target_len)
x = x(:);
if isempty(x)
    y = NaN(target_len, 1);
elseif numel(x) >= target_len
    y = x(1:target_len);
else
    y = [x; repmat(x(end), target_len - numel(x), 1)];
end
end

function cfg = load_experiment_config(repo_root, experiment_name)
config_file = fullfile(repo_root, 'configs', [experiment_name, '.json']);
if exist(config_file, 'file') ~= 2
    config_file = fullfile(repo_root, 'configs', [experiment_name, '.JSON']);
end
assert(exist(config_file, 'file') == 2, 'Could not find experiment config file for experiment "%s".', experiment_name);
cfg = jsondecode(fileread(config_file));
end

function value = get_optional_field(s, name, default_value)
if isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end

function cells = cellstr_value(value)
if iscell(value)
    cells = cellfun(@char, value, 'UniformOutput', false);
elseif isstring(value)
    cells = cellstr(value);
elseif ischar(value)
    cells = cellstr(value);
else
    cells = cellstr(string(value));
end
cells = cells(:).';
end

function result = run_identification_method(method_name, cfg, r, nu, c, theta_true, na, nb, nf, nd, lambda_g, K_max, conv_threshold, enforce_fixed_iterations, best_error_so_far, verbose, metadata)
assert(K_max >= 1 && conv_threshold >= 0, 'Invalid iteration settings.');
init = initialize_method_state(cfg, r, c, na, nb, nf, nd, lambda_g, numel(theta_true));
theta = init.theta_hat;
alpha = init.alpha_hat; e = init.e_hat;
theta0 = theta;
metadata.data_sha256 = hash_values([r(:); c(:)]);
metadata.initialization_sha256 = hash_values(theta0);
metadata.backend = 'matlab'; metadata.dtype = class(r); metadata.device = 'cpu';
metadata.matlab_version = version;
metadata.dataset_id = metadata.data_sha256;
resolved_config_json = jsonencode(cfg);
metadata.config_sha256 = sha256_bytes(unicode2native(resolved_config_json,'UTF-8'));
metadata.run_id = [metadata.dataset_id ':' method_name ':' metadata.initialization_sha256 ':' metadata.config_sha256];
status = 'max_iterations'; status_msg = 'Iteration budget exhausted.';
updates = 0; accepted = 0; stopping_iteration = -1; diagnostic_time = 0;
rows = {}; snapshots = {};
t_start = tic;
Phi = build_state_matrix(alpha,e,r,na,nb,nf,nd);
theta_hist = theta;
param_err = norm(theta-theta_true)/norm(theta_true)*100;
RMSE_hist = sqrt(mean((c-simulate_wiener(r,nu,theta,na,nb,nf,nd,lambda_g)).^2));
est_loss_hist = 0.5*sum((c-Phi*theta).^2);
rel_change = 0; iter_time = 0; cum_time = 0; active_mask = true;
final_update_loss = est_loss_hist(1);
if ~all(isfinite([RMSE_hist;est_loss_hist]))
    status = 'failed'; status_msg = 'Non-finite initial objective or reconstruction.';
end
for k = 1:(K_max-1)
    if ~strcmp(status,'max_iterations'), break; end
    t_iter = tic;
    updates = updates+1;
    try
        [candidate, aux] = method_parameter_update(method_name,cfg,Phi,c,theta,k,nd);
    catch exc
        status = 'failed'; status_msg = exc.message; break;
    end
    if strcmp(aux.status,'not_implemented')
        status = 'skipped'; status_msg = aux.message; break;
    end
    td = tic;
    [row, snapshot] = pl_step_diagnostics(method_name,cfg,Phi,c,theta,candidate,aux,k-1);
    step_diagnostic_time = toc(td);
    diagnostic_time = diagnostic_time+step_diagnostic_time;
    rows{end+1} = row; %#ok<AGROW>
    if cfg.diagnostics.save_matrices, snapshots{end+1} = snapshot; end %#ok<AGROW>
    if ~all(isfinite(candidate))
        rows{end}.rejected = 1; rows{end}.applied_step_norm = 0;
        status = 'invalid_update'; status_msg = 'Rejected non-finite parameters.'; break;
    end
    nu_hat = c-Phi*candidate;
    d = candidate(na+nb+nf:end);
    e_next = filter(1,[1;d].',nu_hat);
    alpha_next = filter([0;candidate(na+1:na+nb)].',[1;candidate(1:na)].',r);
    Phi_next = build_state_matrix(alpha_next,e_next,r,na,nb,nf,nd);
    rmse = sqrt(mean((c-simulate_wiener(r,nu,candidate,na,nb,nf,nd,lambda_g)).^2));
    loss = 0.5*sum((c-Phi_next*candidate).^2);
    if ~all(isfinite([loss;rmse;alpha_next;e_next]))
        rows{end}.rejected = 1; rows{end}.applied_step_norm = 0;
        status = 'invalid_update'; status_msg = 'Rejected non-finite objective or latent state.'; break;
    end
    change = norm(candidate-theta)/(norm(theta)+1e-15);
    final_update_loss = 0.5*sum(nu_hat.^2);
    theta = candidate; accepted = accepted+1;
    if change < conv_threshold
        if contains(method_name,'LGHAM')
            satisfied = abs(final_update_loss-cfg.resolved_epsilon0) <= cfg.stopping.loss_tolerance*max(1,abs(cfg.resolved_epsilon0));
        else
            satisfied = norm(Phi.'*(Phi*theta-c)) <= cfg.stopping.gradient_tolerance;
        end
        status = 'stalled'; status_msg = 'Small step without satisfying the residual tolerance.';
        if satisfied, status = 'converged'; status_msg = 'Small step and residual tolerance satisfied.'; end
        stopping_iteration = k;
    end
    Phi = Phi_next;
    theta_hist(:,end+1) = theta; %#ok<AGROW>
    param_err(end+1,1) = norm(theta-theta_true)/norm(theta_true)*100; %#ok<AGROW>
    RMSE_hist(end+1,1) = rmse; est_loss_hist(end+1,1) = loss; %#ok<AGROW>
    rel_change(end+1,1) = change; active_mask(end+1,1) = true; %#ok<AGROW>
    iter_time(end+1,1) = max(0,toc(t_iter)-step_diagnostic_time); %#ok<AGROW>
    cum_time(end+1,1) = cum_time(end)+iter_time(end); %#ok<AGROW>
end
wall_time = toc(t_start);
valid = ismember(status,{'converged','stalled','max_iterations'});
final_err = NaN;
if valid
    final_err = param_err(end);
    if ~contains(method_name,'LGHAM'), best_error_so_far = min(best_error_so_far,final_update_loss); end
end
if enforce_fixed_iterations && valid
    extra = K_max-size(theta_hist,2);
    theta_hist = [theta_hist repmat(theta,1,extra)];
    param_err = [param_err;repmat(param_err(end),extra,1)];
    RMSE_hist = [RMSE_hist;repmat(RMSE_hist(end),extra,1)];
    est_loss_hist = [est_loss_hist;repmat(est_loss_hist(end),extra,1)];
    rel_change = [rel_change;zeros(extra,1)]; iter_time = [iter_time;zeros(extra,1)];
    cum_time = [cum_time;repmat(cum_time(end),extra,1)];
    active_mask = [active_mask;false(extra,1)];
end
diagnostics = pack_diagnostics(rows);
result = struct('name',method_name,'status',status,'status_msg',status_msg, ...
    'theta_hat',theta,'theta0',theta0,'theta_hist',theta_hist,'param_err',param_err, ...
    'rel_change',rel_change,'RMSE_hist',RMSE_hist,'rmse_hist',RMSE_hist,'est_loss_hist',est_loss_hist, ...
    'iter_time',iter_time,'cum_time',cum_time,'active_mask',active_mask,'final_err',final_err, ...
    'total_time',max(0,wall_time-diagnostic_time),'total_wall_time',wall_time,'diagnostic_time',diagnostic_time, ...
    'iterations',size(theta_hist,2),'update_count',updates,'accepted_updates',accepted, ...
    'stopping_iteration',stopping_iteration,'best_error_so_far',best_error_so_far, ...
    'run_metadata',metadata,'resolved_config_json',resolved_config_json, ...
    'objective_definition','half_sum_squared_refreshed_PL_residual', ...
    'rmse_definition','true_innovation_reconstruction','diagnostics',diagnostics, ...
    'diagnostic_summary',summarize_optimization_diagnostics(diagnostics),'derivative_snapshots',{snapshots});
if verbose, fprintf('%s: %s; %d accepted updates. %s\n',method_name,status,accepted,status_msg); end
end

function [row, snapshot] = pl_step_diagnostics(name,cfg,Phi,c,theta,candidate,aux,iteration)
row = struct('evaluation_iteration',iteration,'available',0,'diagnostic_status','ok', ...
    'solver_kind','none','projected',aux.projected,'rejected',0, ...
    'solver_fallback',aux.solver_fallback,'proposed_step_norm',aux.proposed_step_norm, ...
    'applied_step_norm',norm(candidate-theta),'d_step_multiplier',get_optional_field(cfg,'d_step_multiplier',1));
snapshot = struct('evaluation_iteration',iteration);
if ~cfg.diagnostics.enabled, row.diagnostic_status = 'disabled'; return; end
g = Phi.'*(Phi*theta-c);
row.gradient_norm = norm(g);
row.update_loss_before = 0.5*sum((c-Phi*theta).^2);
row.update_loss_after = 0.5*sum((c-Phi*candidate).^2);
if strcmpi(name,'WS-GGI'), return; end
H = Phi.'*Phi;
try
    row = add_matrix_metrics(row,H,'raw'); row.available = 1;
    reg = max(0,get_optional_field(cfg,'hessian_regularization',0));
    solver = [];
    if strcmpi(name,'WS-GNI')
        solver = H+reg*eye(size(H)); row.solver_kind = 'regularized_hessian';
    elseif contains(upper(name),'DH')
        solver = diag(sum(Phi.^2,1).'+reg); row.solver_kind = 'regularized_diagonal';
    elseif strcmpi(name,'RGLS')
        reg = 0;
        if isfield(cfg,'regularization') && get_optional_field(cfg.regularization,'enabled',false)
            reg = max(0,get_optional_field(cfg.regularization,'lambda',0));
        end
        if reg > 0
            solver = H+reg*eye(size(H)); row.solver_kind = 'regularized_normal_matrix';
        else
            row.solver_kind = 'rectangular_least_squares';
            row.design_condition = cond(Phi); % Actual solve is Phi\c, not H\(Phi'*c).
        end
    elseif contains(name,'LGHAM')
        row.solver_kind = 'gradient_row';
        reg = get_optional_field(cfg,'tikhonov_lambda',0);
        if ~strcmpi(get_optional_field(cfg,'inverse_solver','pinv'),'tikhonov'), reg = 0; end
        row.inverse_denominator = dot(g,g)+reg^2; % MATLAB lsqminnorm uses lambda^2.
    end
    row.regularization = reg;
    if ~isempty(solver), row = add_matrix_metrics(row,solver,'solver'); end
    if isfield(cfg.diagnostics,'parameter_scales')
        scales = cfg.diagnostics.parameter_scales(:);
        assert(numel(scales)==numel(theta) && all(isfinite(scales) & scales>0),'Invalid parameter_scales.');
        metrics = matrix_diagnostics(diag(scales)*H*diag(scales));
        row.scaled_condition = metrics.condition;
    end
    if cfg.diagnostics.save_matrices
        snapshot.hess = H; snapshot.solver_matrix = solver;
    end
catch exc
    row.diagnostic_status = exc.message;
end
end

function row = add_matrix_metrics(row,A,prefix)
metrics = matrix_diagnostics(A);
keys = fieldnames(metrics);
for j = 1:numel(keys), row.([prefix '_' keys{j}]) = metrics.(keys{j}); end
end

function d = pack_diagnostics(rows)
d = struct('evaluation_iteration',zeros(0,1));
if isempty(rows), return; end
keys = {};
for j = 1:numel(rows), keys = union(keys,fieldnames(rows{j})); end
for i = 1:numel(keys)
    key = keys{i};
    is_text = ismember(key,{'solver_kind','diagnostic_status'});
    if is_text, values = repmat({''},numel(rows),1); else, values = NaN(numel(rows),1); end
    for j = 1:numel(rows)
        if isfield(rows{j},key)
            if is_text, values{j} = rows{j}.(key); else, values(j) = rows{j}.(key); end
        end
    end
    d.(key) = values;
end
end

function digest = hash_values(values)
values = double(values(:));
[~,~,endian] = computer;
if endian == 'B', values = swapbytes(values); end
digest = sha256_bytes(typecast(values,'uint8'));
end

function digest = sha256_bytes(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(bytes));
digest = lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function [theta_new, aux_state] = method_parameter_update(method_name, method_cfg, Phi_hat, c, theta_hat, ~, nd)
theta_new = theta_hat;
aux_state = struct('status', 'ok', 'message', '', 'projected',false,'solver_fallback',0,'proposed_step_norm',NaN);

switch upper(method_name)
    case 'WS-GGI'
        if ~isfield(method_cfg, 'step_scale')
            error('WS-GGI config requires field "step_scale".');
        end
        delta = method_cfg.step_scale / max(norm(Phi_hat, 'fro')^2, eps);
        update_dir = delta * (Phi_hat' * (c - Phi_hat * theta_hat));
        d_scale = build_d_step_scale_vector(method_cfg, numel(theta_hat), nd);
        theta_new = theta_hat + d_scale .* update_dir;

    case 'RGLS'
        raw = solve_stable_least_squares(Phi_hat,c,method_cfg);
        theta_new = raw;
        if all(isfinite(raw)), theta_new = project_stable_theta(raw,method_cfg); end
        aux_state.projected = ~isequal(raw,theta_new);
        aux_state.proposed_step_norm = norm(raw-theta_hat);

    case 'WS-GNI'
        [theta_new, aux_state.solver_fallback] = ws_gni_update(Phi_hat, c, theta_hat, method_cfg, nd);

    case {'WS_GGHAM_1_DH', 'WS-GGHAM-1-DH'}
        theta_new = ws_ggham_1_dh_update(Phi_hat, c, theta_hat, method_cfg);

    case 'WS-GGHAM-2-I'
        theta_new = ws_ggham_2_i_update(Phi_hat, c, theta_hat, method_cfg);

    case {'WS_GGHAM_2_DH', 'WS-GGHAM-2-DH'}
        theta_new = ws_ggham_2_dh_update(Phi_hat, c, theta_hat, method_cfg);

    case {'WS-LGHAM-1', 'WS_LGHAM_1', 'WS-LGHAM-1-PINV', 'WS_LGHAM_1_PINV', 'WS-LGHAM-1-TIK', 'WS_LGHAM_1_TIK'}
        theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, get_method_order(method_cfg, 1));

    case {'WS-LGHAM-2', 'WS_LGHAM_2','WS-LGHAM-2-TIK', 'WS_LGHAM_2_TIK'}
        theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, get_method_order(method_cfg, 2));

    case {'WS-LGHAM-3', 'WS_LGHAM_3','WS-LGHAM-3-TIK', 'WS_LGHAM_3_TIK'}
        theta_new = ws_lgham_update(Phi_hat, c, theta_hat, method_cfg, get_method_order(method_cfg, 3));

    case {'WS-GGHAM-1', 'WS-GGHAM-2'}
        aux_state.status = 'not_implemented';
        aux_state.message = sprintf('%s placeholder selected. Add its theta-update rule in method_parameter_update().', method_name);

    otherwise
        aux_state.status = 'not_implemented';
        aux_state.message = sprintf('Unknown method "%s". Add its config and update rule before running it.', method_name);
end
if isnan(aux_state.proposed_step_norm), aux_state.proposed_step_norm = norm(theta_new-theta_hat); end
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

%theta_1 = eta * solve_loss_root_inverse(gradient_root, J_theta - eps0, method_cfg);
theta_1 = eta * solve_loss_root_inverse(gradient_root, J_theta - eps0, method_cfg);
theta_new = theta_hat + theta_1;

if order >= 2
    theta_2 = theta_1 + eta * solve_loss_root_inverse(gradient_root, gradient_root.' * theta_1, method_cfg);
    theta_new = theta_new + theta_2;
end

if order >= 3
    rhs_3 = 0.5 * (theta_1.' * H * theta_1) + gradient_root.' * theta_2;
    theta_3 = theta_2 + eta * solve_loss_root_inverse(gradient_root, rhs_3, method_cfg);
    theta_new = theta_new + theta_3;
end
end

function eps0 = get_eps0(method_cfg)
eps0 = 0;
if isfield(method_cfg, 'eps_0')
    eps0 = method_cfg.eps_0;
elseif isfield(method_cfg, 'epsilon0')
    eps0 = method_cfg.epsilon0;
end
if strcmp(eps0,"automatic")
    eps0 = method_cfg.best_error_so_far*1; % HARDCODED MULTIPLIER
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
            direction = zeros(size(gradient_root));
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

function [theta_new, fallback] = ws_gni_update(Phi_hat, c, theta_hat, method_cfg, nd)
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

fallback = rcond(H) < 1e-12;
if fallback
    direction = pinv(H) * gradient;
else
    direction = H \ gradient;
end

step_scale = build_d_step_scale_vector(method_cfg, numel(theta_hat), nd);
theta_new = theta_hat + step_size * (step_scale .* direction);
end

function step_scale = build_d_step_scale_vector(method_cfg, n_params, nd)
step_scale = ones(n_params, 1);

if nd <= 0
    return;
end

d_multiplier = 1.0;
if isfield(method_cfg, 'd_step_multiplier')
    d_multiplier = method_cfg.d_step_multiplier;
end

if d_multiplier == 1.0
    return;
end

step_scale((n_params - nd + 1):n_params) = d_multiplier;
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

function init_state = initialize_method_state(method_cfg, r, c, na, nb, nf, nd, lambda_g, n)
alpha_hat = 1e-6 * ones(lambda_g, 1);
e_hat = 1e-6 * ones(lambda_g, 1);
theta_hat = 1e-6 * ones(n, 1);

if ~isfield(method_cfg, 'initialization')
    init_state = struct('alpha_hat', alpha_hat, 'e_hat', e_hat, 'theta_hat', theta_hat);
    return;
end

init_cfg = method_cfg.initialization;
policy = get_optional_field(init_cfg,'policy','legacy');
if strcmpi(policy,'consistent')
    theta_hat = init_cfg.theta0(:);
    assert(numel(theta_hat)==n && all(isfinite(theta_hat)), 'theta0 must contain n finite parameters.');
    a = theta_hat(1:na); d = theta_hat(na+nb+nf:end);
    assert(all(abs(roots([1;a]))<1) && all(abs(roots([1;d]))<1), 'theta0 denominators must be stable.');
    alpha_hat = filter([0;theta_hat(na+1:na+nb)].',[1;a].',r);
    beta = polyval([flipud(theta_hat(na+nb+1:na+nb+nf-1));1;0].',alpha_hat);
    e_hat = c-beta;
    init_state = struct('alpha_hat',alpha_hat,'e_hat',e_hat,'theta_hat',theta_hat);
    return;
end
assert(strcmpi(policy,'legacy'),'Unknown initialization policy.');

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

function tf = has_method_config(configs, method_name)
field_name = matlab.lang.makeValidName(method_name);
tf = isfield(configs, field_name);
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

function payload = save_comparison_results(output_file, results_in, selected_methods, na, nb, nf, nd, theta_true, param_names, param_labels, lambda_g, K_max, conv_threshold, sigma_nu, burn_in, enforce_fixed_iterations, experiment_name, mode, seed, N_MC, example_data, mc_summary, all_runs, resolved_experiment)
out_dir = fileparts(output_file);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

parametrization = 'PL';
generated_at = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
experiment = struct( ...
    'name', experiment_name, ...
    'mode', mode, ...
    'seed', seed, ...
    'N_MC', N_MC, ...
    'lambda_g', lambda_g, ...
    'K_max', K_max, ...
    'conv_threshold', conv_threshold, ...
    'sigma_nu', sigma_nu, ...
    'burn_in', burn_in, ...
    'enforce_fixed_iterations', logical(enforce_fixed_iterations), ...
    'dims', struct('na', na, 'nb', nb, 'nf', nf, 'nd', nd), ...
    'theta_true', theta_true(:), ...
    'param_names', {param_names}, ...
    'param_labels', {param_labels}, ...
    'selected_methods', {selected_methods});

results = results_in;
schema_version = 2;
resolved_experiment_json = jsonencode(resolved_experiment);
payload = struct('parametrization',parametrization,'generated_at',generated_at, ...
    'experiment',experiment,'results',results,'example_data',example_data,'mc_summary',mc_summary, ...
    'all_runs',{all_runs},'schema_version',schema_version,'resolved_experiment_json',resolved_experiment_json);
save(output_file,'-struct','payload','-v7');
fprintf('\nSaved PL results to: %s\n', output_file);
end

function plot_method_metrics(results)
valid_idx = find(ismember({results.status}, {'ok','converged','stalled','max_iterations'}));
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
valid_idx = find(ismember({results.status}, {'ok','converged','stalled','max_iterations'}));
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
