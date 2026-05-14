%% Plot saved Wiener identification comparison results.
%
% Run python/comparison_torch.py and mlab/comparison.m first to generate:
%   results/comparison_fsm.mat
%   results/comparison_pl.mat

clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..');
experiment_name = 'example_CSTR';
mode = 'EXAMPLE';  % EXAMPLE or MONTECARLO
data_files = { ...
    fullfile(repo_root, 'results', experiment_name, sprintf('comparison_fsm_%s.mat', lower(mode))), ...
    fullfile(repo_root, 'results', experiment_name, sprintf('comparison_pl_%s.mat', lower(mode))) ...
};
export_figures = true;
fig_dir = fullfile(repo_root, 'figs');

set(groot, ...
    'defaultTextInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...
    'defaultLegendInterpreter', 'latex', ...
    'defaultAxesFontName', 'Times New Roman', ...
    'defaultTextFontName', 'Times New Roman', ...
    'defaultAxesFontSize', 11, ...
    'defaultLineLineWidth', 1.55);

datasets = struct([]);
for i = 1:numel(data_files)
    if exist(data_files{i}, 'file') ~= 2
        warning('Result file not found: %s', data_files{i});
        continue;
    end
    loaded = load(data_files{i});
    ds = normalize_loaded_dataset(loaded, data_files{i});
    if isempty(datasets)
        datasets = ds;
    else
        datasets(end + 1) = ds; %#ok<SAGROW>
    end
end

if isempty(datasets)
    error('No result files were loaded. Run the experiment scripts first.');
end

print_wiener_parameters(datasets);
validate_common_wiener_system(datasets);
print_result_tables(datasets);

method_names = collect_method_names(datasets);
colors = build_distinct_colors(max(numel(method_names), 1));
experiment_tag = lower(sanitize_filename(experiment_name));
mode_tag = lower(sanitize_filename(mode));

for i = 1:numel(datasets)
    parametrization_tag = lower(sanitize_filename(datasets(i).parametrization));

    fig_metrics = plot_metric_summary(datasets(i), method_names, colors);
    metrics_filename = sprintf('comparison_metrics_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_metrics, fig_dir, metrics_filename, export_figures);

    fig_params = plot_parameter_trajectories_saved(datasets(i), method_names, colors);
    filename = sprintf('parameter_trajectories_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_params, fig_dir, filename, export_figures);

    fig_output = plot_output_signals(datasets(i), method_names, colors, 300);
    output_filename = sprintf('output_signals_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_output, fig_dir, output_filename, export_figures);

    if isfield(datasets(i), 'mc_summary') && ~isempty(datasets(i).mc_summary)
        fig_mc = plot_montecarlo_summary(datasets(i), method_names, colors);
        mc_filename = sprintf('montecarlo_summary_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
        export_if_requested(fig_mc, fig_dir, mc_filename, export_figures);

        fig_box = plot_montecarlo_boxplots(datasets(i));
        box_filename = sprintf('montecarlo_parameter_boxplots_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
        export_if_requested(fig_box, fig_dir, box_filename, export_figures);

        fig_err_box = plot_montecarlo_final_error_boxplots(datasets(i));
        err_box_filename = sprintf('montecarlo_final_error_boxplots_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
        export_if_requested(fig_err_box, fig_dir, err_box_filename, export_figures);
    end
end

fprintf('\nDone. MATLAB figures rendered from saved comparison data.\n');


function ds = normalize_loaded_dataset(loaded, file_name)
ds = struct();
ds.file = file_name;
ds.parametrization = char_value(loaded.parametrization);
ds.generated_at = char_value(loaded.generated_at);
ds.experiment = loaded.experiment;
if isfield(loaded, 'example_data')
    ds.example_data = loaded.example_data;
else
    ds.example_data = [];
end
if isfield(loaded, 'mc_summary')
    ds.mc_summary = loaded.mc_summary;
else
    ds.mc_summary = [];
end

raw_results = loaded.results;
if iscell(raw_results)
    raw_results = [raw_results{:}];
end
ds.results = raw_results(:).';
end

function fig = plot_montecarlo_summary(dataset, method_names, colors)
parametrization = dataset.parametrization;
fig = figure('Name', sprintf('%s Monte Carlo Summary', parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 22 9.5]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl);
plot_mc_axis(ax1, dataset, method_names, colors, 'param_err_mean', 'param_err_stderr');
set(ax1, 'YScale', 'log');
title(ax1, 'Mean parameter error');
ylabel(ax1, '$e_\theta$ (\%)');

ax2 = nexttile(tl);
plot_mc_axis(ax2, dataset, method_names, colors, 'rmse_mean', 'rmse_stderr');
set(ax2, 'YScale', 'log');
title(ax2, 'Mean output RMSE');
ylabel(ax2, 'RMSE');

legend(ax2, 'Location', 'best', 'FontSize', 8);
end

function plot_mc_axis(ax, dataset, method_names, colors, mean_field, stderr_field)
hold(ax, 'on');
summary = dataset.mc_summary;
mc_methods = cellstr_value(summary.method_names);
y_mean_all = double(summary.(mean_field));
y_stderr_all = double(summary.(stderr_field));
if size(y_mean_all, 1) ~= numel(mc_methods)
    y_mean_all = y_mean_all.';
    y_stderr_all = y_stderr_all.';
end

for m = 1:numel(mc_methods)
    if ~is_plottable_method(dataset, mc_methods{m})
        continue;
    end
    color_idx = find(strcmp(method_names, mc_methods{m}), 1, 'first');
    if isempty(color_idx)
        continue;
    end
    y_mean = max(y_mean_all(m, :), realmin('double'));
    y_stderr = y_stderr_all(m, :);
    x = 0:(numel(y_mean) - 1);
    lo = max(y_mean - y_stderr, realmin('double'));
    hi = y_mean + y_stderr;
    fill(ax, [x fliplr(x)], [lo fliplr(hi)], colors(color_idx, :), ...
        'FaceAlpha', 0.14, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(ax, x, y_mean, 'Color', colors(color_idx, :), ...
        'DisplayName', latex_label(display_method_label(mc_methods{m}, dataset.parametrization)));
end
grid(ax, 'on');
box(ax, 'on');
xlabel(ax, 'Iteration');
end

function fig = plot_montecarlo_boxplots(dataset)
summary = dataset.mc_summary;
theta_true = double(dataset.experiment.theta_true(:));
param_labels = cellstr_value(dataset.experiment.param_labels);
method_names = cellstr_value(summary.method_names);
errors = double(summary.final_param_errors);
keep_idx = plottable_method_indices(dataset, method_names);
method_names = method_names(keep_idx);
method_labels = display_method_labels(method_names, dataset.parametrization);
errors = errors(keep_idx, :, :);
n_params = numel(theta_true);
n_cols = 3;
n_rows = ceil(n_params / n_cols);

fig = figure('Name', sprintf('%s Monte Carlo Parameter Errors', dataset.parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [3 3 22 6.4 * n_rows]);
tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for p = 1:n_params
    ax = nexttile(tl);
    values = squeeze(errors(:, :, p)).';
    boxplot(ax, values, 'Labels', method_labels, 'LabelOrientation', 'inline');
    yline(ax, 0, 'k--', 'LineWidth', 1.0);
    grid(ax, 'on');
    box(ax, 'on');
    ylabel(ax, '$\hat{\theta}-\theta^\star$');
    if p <= numel(param_labels)
        title(ax, sprintf('$%s$ final error', param_labels{p}));
    else
        title(ax, sprintf('$\\theta_%d$ final error', p));
    end
    ylim(ax, robust_axis_limits(values(:), 0));
end

for p = (n_params + 1):(n_rows * n_cols)
    axis(nexttile(tl), 'off');
end

title(tl, sprintf('%s Parametrization: Monte Carlo Final Parameter Errors', dataset.parametrization), 'FontWeight', 'bold', 'FontSize', 11);
end

function fig = plot_montecarlo_final_error_boxplots(dataset)
summary = dataset.mc_summary;
method_names = cellstr_value(summary.method_names);
final_errors = double(summary.final_errors);
keep_idx = plottable_method_indices(dataset, method_names);
method_names = method_names(keep_idx);
method_labels = display_method_labels(method_names, dataset.parametrization);
final_errors = final_errors(keep_idx, :).';

fig = figure('Name', sprintf('%s Monte Carlo Final Parameter Error', dataset.parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [4 4 22 9]);
ax = axes(fig);
boxplot(ax, final_errors, 'Labels', method_labels, 'LabelOrientation', 'inline');
grid(ax, 'on');
box(ax, 'on');
ylabel(ax, '$e_\theta$ (\%)');
title(ax, sprintf('%s Parametrization: Monte Carlo Final Parameter Error', dataset.parametrization), ...
    'FontWeight', 'bold');
end

function print_wiener_parameters(datasets)
fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('WIENER SYSTEM PARAMETERS IN SAVED RESULTS\n');
fprintf('%s\n', repmat('=', 1, 72));
for i = 1:numel(datasets)
    exp_cfg = datasets(i).experiment;
    dims = exp_cfg.dims;
    theta_true = double(exp_cfg.theta_true(:));
    names = cellstr_value(exp_cfg.param_names);

    fprintf('\n%s parametrization (%s)\n', datasets(i).parametrization, datasets(i).file);
    fprintf('  na = %d, nb = %d, nf = %d, nd = %d\n', ...
        scalar_value(dims.na), scalar_value(dims.nb), scalar_value(dims.nf), scalar_value(dims.nd));
    fprintf('  lambda_g = %d, sigma_nu = %.6g, burn_in = %d\n', ...
        scalar_value(exp_cfg.lambda_g), scalar_value(exp_cfg.sigma_nu), scalar_value(exp_cfg.burn_in));
    for p = 1:numel(theta_true)
        if p <= numel(names)
            name = names{p};
        else
            name = sprintf('theta_%d', p);
        end
        fprintf('  %-4s = %.10g\n', name, theta_true(p));
    end
end
end

function print_result_tables(datasets)
for i = 1:numel(datasets)
    fprintf('\n%s\n', repmat('=', 1, 108));
    fprintf('FINAL METHOD SUMMARY (%s parametrization)\n', datasets(i).parametrization);
    fprintf('%s\n', repmat('=', 1, 108));
    fprintf('%-18s %-12s %-14s %-14s %-12s %-12s\n', ...
        'Method', 'Status', 'Param err(%)', 'Output RMSE', 'Final d1', 'CPU time (s)');
    fprintf('%s\n', repmat('-', 1, 108));

    for m = 1:numel(datasets(i).results)
        result = datasets(i).results(m);
        theta_hat = double(result.theta_hat(:));
        final_d = NaN;
        if ~isempty(theta_hat)
            final_d = theta_hat(end);
        end
        fprintf('%-18s %-12s %-14.5f %-14.5f %-12.5f %-12.5f\n', ...
            display_method_label(result_name(result), datasets(i).parametrization), char_value(result.status), ...
            scalar_value(result.final_err), final_output_rmse(result), final_d, scalar_value(result.total_time));
    end
end
end

function validate_common_wiener_system(datasets)
ref = datasets(1).experiment;
ref_dims = dims_vector(ref.dims);
ref_theta = double(ref.theta_true(:));
ref_lambda = scalar_value(ref.lambda_g);
ref_sigma = scalar_value(ref.sigma_nu);
ref_burn = scalar_value(ref.burn_in);

for i = 2:numel(datasets)
    exp_cfg = datasets(i).experiment;
    same_dims = isequal(ref_dims, dims_vector(exp_cfg.dims));
    same_theta = isequal(size(ref_theta), size(double(exp_cfg.theta_true(:)))) && ...
        all(abs(ref_theta - double(exp_cfg.theta_true(:))) < 1e-12);
    same_data_settings = ref_lambda == scalar_value(exp_cfg.lambda_g) && ...
        abs(ref_sigma - scalar_value(exp_cfg.sigma_nu)) < 1e-12 && ...
        ref_burn == scalar_value(exp_cfg.burn_in);

    if ~(same_dims && same_theta && same_data_settings)
        warning(['Saved datasets do not use the same Wiener system/settings. ', ...
            'Compare %s with %s carefully.'], datasets(1).parametrization, datasets(i).parametrization);
    end
end
end

function methods = collect_method_names(datasets)
methods = {};
for i = 1:numel(datasets)
    for m = 1:numel(datasets(i).results)
        if is_plotted_result(datasets(i).results(m))
            methods{end + 1} = result_name(datasets(i).results(m)); %#ok<AGROW>
        end
    end
end
methods = unique(methods, 'stable');
end

function fig = plot_metric_summary(dataset, method_names, colors)
parametrization = dataset.parametrization;
fig = figure('Name', sprintf('%s Saved Comparison Metrics', parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 22 16.5]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl);
plot_metric_axis(ax1, dataset, method_names, colors, 'param_err', 'iteration');
title(ax1, 'Parameter error vs. iteration');
ylabel(ax1, '$e_\theta$ (\%)');

ax2 = nexttile(tl);
plot_metric_axis(ax2, dataset, method_names, colors, 'param_err', 'time');
title(ax2, 'Parameter error vs. time');
ylabel(ax2, '$e_\theta$ (\%)');

ax3 = nexttile(tl);
plot_metric_axis(ax3, dataset, method_names, colors, 'rmse_hist', 'iteration');
title(ax3, 'Output RMSE vs. iteration');
ylabel(ax3, 'RMSE');

ax4 = nexttile(tl);
plot_metric_axis(ax4, dataset, method_names, colors, 'rmse_hist', 'time');
title(ax4, 'Output RMSE vs. time');
ylabel(ax4, 'RMSE');

legend(ax1, 'Location', 'best', 'FontSize', 8);
title(tl, sprintf('Wiener System Identification: %s Parametrization', parametrization), 'FontWeight', 'bold', 'FontSize', 11);
end

function plot_metric_axis(ax, dataset, method_names, colors, field_name, x_mode)
hold(ax, 'on');
for m = 1:numel(dataset.results)
    result = dataset.results(m);
    if ~is_plotted_result(result)
        continue;
    end

    method = result_name(result);
    color_idx = find(strcmp(method_names, method), 1, 'first');
    if isempty(color_idx)
        continue;
    end
    y = max(result_vector(result, field_name), realmin('double'));
    if strcmp(x_mode, 'time')
        x = result_vector(result, 'cum_time');
        xlabel(ax, 'Compute time (s)');
    else
        x = 0:(numel(y) - 1);
        xlabel(ax, 'Iteration');
    end

    n = min(numel(x), numel(y));
    plot(ax, x(1:n), y(1:n), '-', 'Color', colors(color_idx, :), ...
        'DisplayName', latex_label(display_method_label(method, dataset.parametrization)));
end
set(ax, 'YScale', 'log');
grid(ax, 'on');
box(ax, 'on');
end

function fig = plot_parameter_trajectories_saved(dataset, method_names, colors)
theta_true = double(dataset.experiment.theta_true(:));
param_labels = cellstr_value(dataset.experiment.param_labels);
n_params = numel(theta_true);
n_cols = 3;
n_rows = ceil(n_params / n_cols);

fig = figure('Name', sprintf('%s Parameter Trajectories', dataset.parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [3 3 25 6.4 * n_rows]);
tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
legend_ax = [];

for p = 1:n_params
    ax = nexttile(tl);
    hold(ax, 'on');
    for m = 1:numel(dataset.results)
        result = dataset.results(m);
        if ~is_plotted_result(result)
            continue;
        end
        method = result_name(result);
        color_idx = find(strcmp(method_names, method), 1, 'first');
        if isempty(color_idx)
            continue;
        end
        theta_hist = double(result.theta_hist);
        iter_axis = 0:(size(theta_hist, 2) - 1);
        param_values = theta_hist(p, :);
        plot(ax, iter_axis, param_values, 'Color', colors(color_idx, :), ...
            'DisplayName', latex_label(display_method_label(method, dataset.parametrization)));
    end
    yline(ax, theta_true(p), 'k--', 'LineWidth', 1.1, 'DisplayName', '$\theta^\star$');
    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'Iteration');
    if p <= numel(param_labels)
        label = param_labels{p};
    else
        label = sprintf('\\theta_%d', p);
    end
    ylabel(ax, sprintf('$\\hat{%s}$', label));
    title(ax, sprintf('$%s$ trajectory', label));
    if p == min(3, n_params)
        legend_ax = ax;
    end
end

for p = (n_params + 1):(n_rows * n_cols)
    axis(nexttile(tl), 'off');
end

if ~isempty(legend_ax)
    legend(legend_ax, 'Location', 'northwest', 'FontSize', 8);
end

end

function fig = plot_output_signals(dataset, method_names, colors, zoom_samples)
parametrization = dataset.parametrization;

if isempty(dataset.example_data) || ~all(isfield(dataset.example_data, {'r', 'nu', 'c'}))
    fig = figure('Name', sprintf('%s Output Signals', parametrization), ...
        'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 24 9.5]);
    ax = axes(fig);
    text(ax, 0.5, 0.5, 'No saved output signal data available.', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis(ax, 'off');
    return;
end

r = double(dataset.example_data.r(:));
nu = double(dataset.example_data.nu(:));
c = double(dataset.example_data.c(:));
n_samples = min([zoom_samples, numel(r), numel(nu), numel(c)]);
if n_samples == 0
    fig = figure('Name', sprintf('%s Output Signals', parametrization), ...
        'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 24 9.5]);
    ax = axes(fig);
    text(ax, 0.5, 0.5, 'Saved output signal data is empty.', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis(ax, 'off');
    return;
end

x = 1:n_samples;
dims = dataset.experiment.dims;
na = scalar_value(dims.na);
nb = scalar_value(dims.nb);
nf = scalar_value(dims.nf);
nd = scalar_value(dims.nd);

estimate_signals = {};
estimate_labels = {};
estimate_colors = [];
for m = 1:numel(dataset.results)
    result = dataset.results(m);
    if ~is_plotted_result(result) || isempty(result.theta_hat)
        continue;
    end

    method = result_name(result);
    color_idx = find(strcmp(method_names, method), 1, 'first');
    if isempty(color_idx)
        continue;
    end

    c_est = simulate_wiener(r, nu, double(result.theta_hat(:)), na, nb, nf, nd, numel(c));
    if isempty(c_est) || ~any(isfinite(c_est))
        continue;
    end

    estimate_signals{end + 1} = c_est(1:n_samples); %#ok<AGROW>
    estimate_labels{end + 1} = display_method_label(method, dataset.parametrization); %#ok<AGROW>
    estimate_colors(end + 1, :) = colors(color_idx, :); %#ok<AGROW>
end

n_estimates = numel(estimate_signals);
fig_height = max(9.5, 2.25 * max(n_estimates, 1) + 1.2);
fig = figure('Name', sprintf('%s Output Signals', parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 24 fig_height]);

if n_estimates == 0
    ax = axes(fig);
    text(ax, 0.5, 0.5, 'No plottable estimated output signals available.', ...
        'HorizontalAlignment', 'center', 'Units', 'normalized');
    axis(ax, 'off');
    return;
end

tl = tiledlayout(fig, n_estimates, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
axes_list = gobjects(n_estimates, 1);
measured_color = ones(1,3)*0.3;

for idx = 1:n_estimates
    ax = nexttile(tl);
    axes_list(idx) = ax;
    hold(ax, 'on');
    plot(ax, x, c(1:n_samples), 'Color', measured_color, ...
        'LineWidth', 1.1, 'DisplayName', 'Measured output');
    plot(ax, x, estimate_signals{idx}, '.', ...
        'Color', estimate_colors(idx, :), ...
        'MarkerSize', 5.5, ...
        'LineWidth', 0.9, ...
        'DisplayName', latex_label(estimate_labels{idx}));
    grid(ax, 'on');
    box(ax, 'on');
    xlim(ax, [1 n_samples]);
    minmax = [min(c(1:n_samples)), max(c(1:n_samples))];
    margin = 0.1*max(abs(minmax));
    ylim(ax, [minmax(1)-margin, minmax(2)+margin])
    ylabel(ax, '$c(t)$');
    title(ax, latex_label(estimate_labels{idx}), 'FontWeight', 'normal');
    if idx == 1
        %legend(ax, 'Location', 'best', 'FontSize', 8);
    end
    if idx < n_estimates
        set(ax, 'XTickLabel', []);
    else
        xlabel(ax, 'Sample $t$');
    end
end

linkaxes(axes_list, 'x');
end

function tf = is_ok_result(result)
tf = strcmpi(strtrim(char_value(result.status)), 'ok');
end

function tf = is_plotted_result(result)
tf = is_ok_result(result) && result_cpu_time(result) > 0;
end

function tf = is_plottable_method(dataset, method_name)
tf = false;
for i = 1:numel(dataset.results)
    if strcmp(result_name(dataset.results(i)), method_name)
        tf = is_plotted_result(dataset.results(i));
        return;
    end
end
end

function idx = plottable_method_indices(dataset, method_names)
keep = false(size(method_names));
for i = 1:numel(method_names)
    keep(i) = is_plottable_method(dataset, method_names{i});
end
idx = find(keep);
end

function value = result_cpu_time(result)
if isfield(result, 'total_time')
    value = scalar_value(result.total_time);
elseif isfield(result, 'cum_time') && ~isempty(result.cum_time)
    cum_time = double(result.cum_time(:));
    value = cum_time(end);
else
    value = 0;
end
end

function name = result_name(result)
name = strtrim(char_value(result.name));
end

function value = result_vector(result, field_name)
if isfield(result, field_name)
    value = double(result.(field_name)(:));
elseif strcmp(field_name, 'rmse_hist') && isfield(result, 'RMSE_hist')
    value = double(result.RMSE_hist(:));
else
    error('Result for method %s does not contain field "%s".', result_name(result), field_name);
end
end

function value = final_output_rmse(result)
value = NaN;
if isfield(result, 'rmse_hist')
    values = double(result.rmse_hist(:));
elseif isfield(result, 'RMSE_hist')
    values = double(result.RMSE_hist(:));
else
    return;
end

values = values(isfinite(values));
if ~isempty(values)
    value = values(end);
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

function colors = build_distinct_colors(n)
base = [ ...
    0.000 0.447 0.698;  % blue
    0.835 0.369 0.000;  % vermillion
    0.000 0.620 0.451;  % bluish green
    0.800 0.475 0.655;  % reddish purple
    0.902 0.624 0.000;  % orange
    0.337 0.706 0.914;  % sky blue
    0.600 0.600 0.000;  % olive
    0.494 0.184 0.556;  % deep purple
    0.635 0.078 0.184;  % dark red
    0.301 0.745 0.933;  % cyan
    0.466 0.674 0.188;  % green
    0.850 0.325 0.098]; % orange-red

if n <= size(base, 1)
    colors = base(1:n, :);
    return;
end

extra = hsv(n - size(base, 1) + 1);
colors = [base; extra(1:(n - size(base, 1)), :)];
end

function limits = robust_axis_limits(values, reference_value)
values = double(values(:));
values = values(isfinite(values));
ref = double(reference_value);
if isfinite(ref)
    values = [values; ref];
end

if isempty(values)
    limits = [-1 1];
    return;
end

sorted_values = sort(values);
lo = percentile_value(sorted_values, 5);
hi = percentile_value(sorted_values, 95);
if isfinite(ref)
    lo = min(lo, ref);
    hi = max(hi, ref);
end

if ~isfinite(lo) || ~isfinite(hi)
    limits = [-1 1];
    return;
end

if abs(hi - lo) < eps(max(1, abs(lo)))
    pad = max(1e-3, 0.1 * max(1, abs(lo)));
else
    pad = 0.10 * (hi - lo);
end
limits = [lo - pad, hi + pad];
end

function value = percentile_value(sorted_values, percent)
sorted_values = sorted_values(:);
n = numel(sorted_values);
if n == 1
    value = sorted_values(1);
    return;
end

rank = 1 + (n - 1) * percent / 100;
lower_idx = floor(rank);
upper_idx = ceil(rank);
if lower_idx == upper_idx
    value = sorted_values(lower_idx);
else
    weight = rank - lower_idx;
    value = (1 - weight) * sorted_values(lower_idx) + weight * sorted_values(upper_idx);
end
end

function dims = dims_vector(dims_struct)
dims = [scalar_value(dims_struct.na), scalar_value(dims_struct.nb), ...
        scalar_value(dims_struct.nf), scalar_value(dims_struct.nd)];
end

function value = scalar_value(value_in)
value = double(value_in);
value = value(1);
end

function txt = char_value(value)
if isstring(value)
    txt = char(value);
elseif iscell(value)
    if isempty(value)
        txt = '';
    else
        txt = char_value(value{1});
    end
elseif ischar(value)
    txt = strtrim(value);
else
    txt = char(string(value));
end
end

function cells = cellstr_value(value)
if iscell(value)
    cells = cellfun(@char_value, value, 'UniformOutput', false);
elseif isstring(value)
    cells = cellstr(value);
elseif ischar(value)
    if size(value, 1) == 1
        cells = {strtrim(value)};
    else
        cells = cellstr(value);
    end
else
    cells = {};
end
cells = cells(:).';
end

function labels = display_method_labels(method_names, parametrization)
if nargin < 2
    parametrization = '';
end
labels = cell(size(method_names));
for i = 1:numel(method_names)
    labels{i} = display_method_label(method_names{i}, parametrization);
end
end

function label = display_method_label(method_name, parametrization)
if nargin < 2
    parametrization = '';
end
label = char_value(method_name);
label = strrep(label, 'RGLS', 'RLS');
label = strrep(label, 'GGHAM', 'GHAM');
label = strrep(label, 'LGHAM', 'LHAM');
if contains(label, 'HAM')
    label = regexprep(label, '^WS-', '');
end
if strcmpi(char_value(parametrization), 'FSM')
    if strcmp(label, 'WS-GNI')
        label = 'Newton';
    elseif strcmp(label, 'WS-GGI')
        label = 'GI';
    end
end
end

function label = latex_label(label)
label = strrep(label, '\', '\textbackslash{}');
label = strrep(label, '_', '\_');
label = strrep(label, '%', '\%');
end

function name = sanitize_filename(name)
name = regexprep(name, '[^A-Za-z0-9_-]', '_');
end

function export_if_requested(fig, fig_dir, filename, export_figures)
if ~export_figures
    return;
end
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end
exportgraphics(fig, fullfile(fig_dir, filename), 'ContentType', 'vector');
fprintf('Exported figure: %s\n', fullfile(fig_dir, filename));
end
