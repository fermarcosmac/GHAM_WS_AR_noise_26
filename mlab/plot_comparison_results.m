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
colors = lines(max(numel(method_names), 1));

for i = 1:numel(datasets)
    fig_metrics = plot_metric_summary(datasets(i), method_names, colors);
    metrics_filename = sprintf('comparison_metrics_%s.pdf', lower(sanitize_filename(datasets(i).parametrization)));
    export_if_requested(fig_metrics, fig_dir, metrics_filename, export_figures);

    fig_params = plot_parameter_trajectories_saved(datasets(i), method_names, colors);
    filename = sprintf('parameter_trajectories_%s.pdf', lower(sanitize_filename(datasets(i).parametrization)));
    export_if_requested(fig_params, fig_dir, filename, export_figures);

    if isfield(datasets(i), 'mc_summary') && ~isempty(datasets(i).mc_summary)
        fig_mc = plot_montecarlo_summary(datasets(i), method_names, colors);
        mc_filename = sprintf('montecarlo_summary_%s.pdf', lower(sanitize_filename(datasets(i).parametrization)));
        export_if_requested(fig_mc, fig_dir, mc_filename, export_figures);

        fig_box = plot_montecarlo_boxplots(datasets(i));
        box_filename = sprintf('montecarlo_parameter_boxplots_%s.pdf', lower(sanitize_filename(datasets(i).parametrization)));
        export_if_requested(fig_box, fig_dir, box_filename, export_figures);
    end
end

fprintf('\nDone. MATLAB figures rendered from saved comparison data.\n');


function ds = normalize_loaded_dataset(loaded, file_name)
ds = struct();
ds.file = file_name;
ds.parametrization = char_value(loaded.parametrization);
ds.generated_at = char_value(loaded.generated_at);
ds.experiment = loaded.experiment;
if isfield(loaded, 'mc_summary')
    ds.mc_summary = loaded.mc_summary;
else
    ds.mc_summary = [];
end

raw_results = loaded.results;
if iscell(raw_results)
    raw_results = [raw_results{:}];
end

function fig = plot_montecarlo_summary(dataset, method_names, colors)
summary = dataset.mc_summary;
parametrization = dataset.parametrization;
fig = figure('Name', sprintf('%s Monte Carlo Summary', parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 8]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl);
plot_mc_axis(ax1, summary, method_names, colors, 'param_err_mean', 'param_err_stderr');
set(ax1, 'YScale', 'log');
title(ax1, sprintf('Mean parameter error (%s)', parametrization));
ylabel(ax1, '$e_\theta$ (\%)');

ax2 = nexttile(tl);
plot_mc_axis(ax2, summary, method_names, colors, 'rmse_mean', 'rmse_stderr');
set(ax2, 'YScale', 'log');
title(ax2, sprintf('Mean RMSE (%s)', parametrization));
ylabel(ax2, 'RMSE');

legend(ax1, 'Location', 'best', 'FontSize', 8);
title(tl, sprintf('%s Monte Carlo Curves', parametrization), 'FontWeight', 'bold');
end

function plot_mc_axis(ax, summary, method_names, colors, mean_field, stderr_field)
hold(ax, 'on');
mc_methods = cellstr_value(summary.method_names);
y_mean_all = double(summary.(mean_field));
y_stderr_all = double(summary.(stderr_field));
if size(y_mean_all, 1) ~= numel(mc_methods)
    y_mean_all = y_mean_all.';
    y_stderr_all = y_stderr_all.';
end

for m = 1:numel(mc_methods)
    color_idx = find(strcmp(method_names, mc_methods{m}), 1, 'first');
    if isempty(color_idx)
        color_idx = m;
    end
    y_mean = max(y_mean_all(m, :), realmin('double'));
    y_stderr = y_stderr_all(m, :);
    x = 0:(numel(y_mean) - 1);
    lo = max(y_mean - y_stderr, realmin('double'));
    hi = y_mean + y_stderr;
    fill(ax, [x fliplr(x)], [lo fliplr(hi)], colors(color_idx, :), ...
        'FaceAlpha', 0.14, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(ax, x, y_mean, 'Color', colors(color_idx, :), 'DisplayName', latex_label(mc_methods{m}));
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
n_params = numel(theta_true);
n_cols = 3;
n_rows = ceil(n_params / n_cols);

fig = figure('Name', sprintf('%s Monte Carlo Parameter Errors', dataset.parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [3 3 18 5.2 * n_rows]);
tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for p = 1:n_params
    ax = nexttile(tl);
    values = squeeze(errors(:, :, p)).';
    boxplot(ax, values, 'Labels', method_names, 'LabelOrientation', 'inline');
    yline(ax, 0, 'k--', 'LineWidth', 1.0);
    grid(ax, 'on');
    box(ax, 'on');
    ylabel(ax, '$\hat{\theta}-\theta^\star$');
    if p <= numel(param_labels)
        title(ax, sprintf('$%s$ final error', param_labels{p}));
    else
        title(ax, sprintf('$\\theta_%d$ final error', p));
    end
end

for p = (n_params + 1):(n_rows * n_cols)
    axis(nexttile(tl), 'off');
end

title(tl, sprintf('%s Parametrization: Monte Carlo Final Parameter Errors', dataset.parametrization), 'FontWeight', 'bold');
end
ds.results = raw_results(:).';
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
    fprintf('\n%s\n', repmat('=', 1, 92));
    fprintf('FINAL METHOD SUMMARY (%s parametrization)\n', datasets(i).parametrization);
    fprintf('%s\n', repmat('=', 1, 92));
    fprintf('%-18s %-12s %-12s %-12s %-12s\n', ...
        'Method', 'Status', 'Final err(%)', 'Final d1', 'CPU time (s)');
    fprintf('%s\n', repmat('-', 1, 92));

    for m = 1:numel(datasets(i).results)
        result = datasets(i).results(m);
        theta_hat = double(result.theta_hat(:));
        final_d = NaN;
        if ~isempty(theta_hat)
            final_d = theta_hat(end);
        end
        fprintf('%-18s %-12s %-12.5f %-12.5f %-12.5f\n', ...
            result_name(result), char_value(result.status), ...
            scalar_value(result.final_err), final_d, scalar_value(result.total_time));
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
        if is_ok_result(datasets(i).results(m))
            methods{end + 1} = result_name(datasets(i).results(m)); %#ok<AGROW>
        end
    end
end
methods = unique(methods, 'stable');
end

function fig = plot_metric_summary(dataset, method_names, colors)
parametrization = dataset.parametrization;
fig = figure('Name', sprintf('%s Saved Comparison Metrics', parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 14]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl);
plot_metric_axis(ax1, dataset, method_names, colors, 'param_err', 'iteration');
title(ax1, sprintf('Parameter error vs. iteration (%s)', parametrization));
ylabel(ax1, '$e_\theta$ (\%)');

ax2 = nexttile(tl);
plot_metric_axis(ax2, dataset, method_names, colors, 'param_err', 'time');
title(ax2, sprintf('Parameter error vs. time (%s)', parametrization));
ylabel(ax2, '$e_\theta$ (\%)');

ax3 = nexttile(tl);
plot_metric_axis(ax3, dataset, method_names, colors, 'rmse_hist', 'iteration');
title(ax3, sprintf('RMSE vs. iteration (%s)', parametrization));
ylabel(ax3, 'RMSE');

ax4 = nexttile(tl);
plot_metric_axis(ax4, dataset, method_names, colors, 'rmse_hist', 'time');
title(ax4, sprintf('RMSE vs. time (%s)', parametrization));
ylabel(ax4, 'RMSE');

legend(ax1, 'Location', 'best', 'FontSize', 8);
title(tl, sprintf('Wiener System Identification: %s Parametrization', parametrization), 'FontWeight', 'bold');
end

function plot_metric_axis(ax, dataset, method_names, colors, field_name, x_mode)
hold(ax, 'on');
for m = 1:numel(dataset.results)
    result = dataset.results(m);
    if ~is_ok_result(result)
        continue;
    end

    method = result_name(result);
    color_idx = find(strcmp(method_names, method), 1, 'first');
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
        'DisplayName', latex_label(method));
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
    'Color', 'w', 'Units', 'centimeters', 'Position', [3 3 18 5.2 * n_rows]);
tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for p = 1:n_params
    ax = nexttile(tl);
    hold(ax, 'on');
    for m = 1:numel(dataset.results)
        result = dataset.results(m);
        if ~is_ok_result(result)
            continue;
        end
        method = result_name(result);
        color_idx = find(strcmp(method_names, method), 1, 'first');
        theta_hist = double(result.theta_hist);
        iter_axis = 0:(size(theta_hist, 2) - 1);
        plot(ax, iter_axis, theta_hist(p, :), 'Color', colors(color_idx, :), ...
            'DisplayName', latex_label(method));
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
    if p == 1
        legend(ax, 'Location', 'best', 'FontSize', 8);
    end
end

for p = (n_params + 1):(n_rows * n_cols)
    axis(nexttile(tl), 'off');
end

title(tl, sprintf('%s Parametrization: Parameter Trajectories', dataset.parametrization), 'FontWeight', 'bold');
end

function tf = is_ok_result(result)
tf = strcmpi(strtrim(char_value(result.status)), 'ok');
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
