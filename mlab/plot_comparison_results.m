function datasets = plot_comparison_results(options)
% Optional experiment_name/mode/data_files/study/fig_dir/export_figures overrides.
%% Plot saved Wiener identification comparisons and initialization studies.
% Default selection includes available initialization files for the experiment.
% Use include_initialization=false for comparisons only, or
% study='initialization' for final-estimate boxplots and conditioning figures.

if nargin < 1, options = struct(); end

script_dir = fileparts(mfilename('fullpath'));
repo_root = fullfile(script_dir, '..');
experiment_name = 'example_1';
mode = 'MONTECARLO';  % EXAMPLE, MONTECARLO, or INITIALIZATION
experiment_name = char_value(get_option(options,'experiment_name',experiment_name));
mode = upper(char_value(get_option(options,'mode',mode)));
initialization_files = { ...
    fullfile(repo_root,'results',experiment_name,'initialization_fsm.mat'), ...
    fullfile(repo_root,'results',experiment_name,'initialization_pl.mat')};
if strcmp(mode,'INITIALIZATION')
    data_files = initialization_files;
else
    data_files = { ...
        fullfile(repo_root,'results',experiment_name,sprintf('comparison_fsm_%s.mat',lower(mode))), ...
        fullfile(repo_root,'results',experiment_name,sprintf('comparison_pl_%s.mat',lower(mode)))};
    if ~isfield(options,'data_files') && ~isfield(options,'study') && ...
            get_option(options,'include_initialization',true)
        for k = 1:numel(initialization_files)
            if exist(initialization_files{k},'file')==2
                data_files{end+1} = initialization_files{k}; %#ok<AGROW>
                fprintf('Including initialization results: %s\n',initialization_files{k});
            end
        end
    end
end
export_figures = true;
fig_dir = fullfile(repo_root, 'figs');
if isfield(options,'data_files'), data_files = options.data_files; end
if isfield(options,'study') && ~isfield(options,'data_files')
    experiment_name = char_value(get_option(options,'experiment_name',experiment_name));
    study_name = char_value(options.study);
    assert(ismember(study_name,{'initialization','record_length'}),'Unknown study.');
    data_files = {fullfile(repo_root,'results',experiment_name,[study_name '_fsm.mat']), ...
                  fullfile(repo_root,'results',experiment_name,[study_name '_pl.mat'])};
end
if isfield(options,'fig_dir'), fig_dir = options.fig_dir; end
if isfield(options,'export_figures'), export_figures = options.export_figures; end
diagnostics_only = isfield(options,'diagnostics_only') && options.diagnostics_only;

set(groot, ...
    'defaultTextInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...
    'defaultLegendInterpreter', 'latex', ...
    'defaultAxesFontName', 'Times New Roman', ...
    'defaultTextFontName', 'Times New Roman', ...
    'defaultAxesFontSize', 11, ...
    'defaultLineLineWidth', 1.55);

datasets = struct([]);
record_studies_loaded = false;
for i = 1:numel(data_files)
    if exist(data_files{i}, 'file') ~= 2
        warning('Result file not found: %s', data_files{i});
        continue;
    end
    loaded = load(data_files{i});
    if isfield(loaded,'study') && strcmp(char_value(loaded.study.type),'record_length')
        plot_record_length_study(loaded,struct('fig_dir',fig_dir,'export_figures',export_figures));
        record_studies_loaded = true;
        continue;
    end
    ds = normalize_loaded_dataset(loaded, data_files{i});
    if isempty(datasets)
        datasets = ds;
    else
        datasets(end + 1) = ds; %#ok<SAGROW>
    end
end

if isempty(datasets)
    if record_studies_loaded, return; end
    error('No result files were loaded. Run the experiment scripts first.');
end

print_wiener_parameters(datasets);
validate_common_wiener_system(datasets);
print_result_tables(datasets, mode);

method_names = collect_method_names(datasets);
colors = build_distinct_colors(max(numel(method_names), 1));
experiment_tag = lower(sanitize_filename(experiment_name));
mode_tag = lower(sanitize_filename(mode));

for i = 1:numel(datasets)
    parametrization_tag = lower(sanitize_filename(datasets(i).parametrization));
    if isfield(datasets(i).experiment,'name'), experiment_tag = lower(sanitize_filename(char_value(datasets(i).experiment.name))); end
    if isfield(datasets(i).experiment,'mode'), mode_tag = lower(sanitize_filename(char_value(datasets(i).experiment.mode))); end
    prefix = sprintf('%s_%s_%s',experiment_tag,mode_tag,parametrization_tag);
    plot_optimization_diagnostics(datasets(i),fig_dir,prefix,export_figures,method_names,colors);
    if diagnostics_only, continue; end
    if isfield(datasets(i).study,'type') && strcmp(char_value(datasets(i).study.type),'initialization')
        fig_box = plot_montecarlo_boxplots(datasets(i));
        export_if_requested(fig_box,fig_dir,['initialization_parameter_boxplots_' prefix '.pdf'],export_figures);
        fig_status = plot_initialization_outcomes(datasets(i));
        export_if_requested(fig_status,fig_dir,['initialization_outcomes_' prefix '.pdf'],export_figures);
        continue;
    end

    fig_metrics = plot_metric_summary(datasets(i), method_names, colors);
    metrics_filename = sprintf('comparison_metrics_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_metrics, fig_dir, metrics_filename, export_figures);

    fig_params = plot_parameter_trajectories_saved(datasets(i), method_names, colors);
    filename = sprintf('parameter_trajectories_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_params, fig_dir, filename, export_figures);

    [fig_output, fig_error] = plot_output_signals(datasets(i), method_names, colors, 300);
    output_filename = sprintf('output_signals_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_output, fig_dir, output_filename, export_figures);
    error_filename = sprintf('output_errors_%s_%s_%s.pdf', experiment_tag, mode_tag, parametrization_tag);
    export_if_requested(fig_error, fig_dir, error_filename, export_figures);

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


end

function ds = normalize_loaded_dataset(loaded, file_name)
ds = struct();
ds.file = file_name;
ds.parametrization = char_value(loaded.parametrization);
ds.generated_at = char_value(loaded.generated_at);
ds.experiment = loaded.experiment;
ds.study = struct();
if isfield(loaded,'study'), ds.study = loaded.study; end
% Earlier initialization payloads may identify the study only through mode.
if ~isfield(ds.study,'type') && isfield(ds.experiment,'mode') && ...
        strcmpi(char_value(ds.experiment.mode),'INITIALIZATION')
    ds.study.type = 'initialization';
end
ds.all_runs = {};
if isfield(loaded,'all_runs'), ds.all_runs = loaded.all_runs; end
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
initialization_study = isfield(dataset.study,'type') && strcmp(char_value(dataset.study.type),'initialization');
if initialization_study
    errors = double(summary.final_param_estimates);
else
    errors = double(summary.final_param_errors);
end
keep_idx = plottable_method_indices(dataset, method_names);
if initialization_study, keep_idx = 1:numel(method_names); end
method_names = method_names(keep_idx);
method_labels = display_method_labels(method_names, dataset.parametrization);
errors = errors(keep_idx, :, :);
n_params = numel(theta_true);
n_cols = 3;
n_rows = ceil(n_params / n_cols);

fig = figure('Name', sprintf('%s Monte Carlo Parameter Errors', dataset.parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [3 3 22 6.4 * n_rows]);
if initialization_study
    fig.Position=[3 3 30 8.5*n_rows];
    fig.Name=[dataset.parametrization ' initialization final parameter estimates'];
end
tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

for p = 1:n_params
    ax = nexttile(tl);
    values = reshape(errors(:,:,p),numel(method_names),[]).';
    if any(isfinite(values),'all')
        if initialization_study
            groups=repmat(1:numel(method_names),size(values,1),1);
            keep=isfinite(values);
            boxchart(ax,groups(keep),values(keep),'BoxFaceColor',[.2 .45 .7]);
        else
            boxplot(ax, values, 'Labels', method_labels, 'LabelOrientation', 'inline');
        end
    else
        text(ax,.5,.5,'No finite estimates','Units','normalized','HorizontalAlignment','center');
        xticks(ax,1:numel(method_labels)); xticklabels(ax,method_labels);
    end
    if initialization_study
        xticks(ax,1:numel(method_labels)); xticklabels(ax,method_labels); xtickangle(ax,40);
        xlim(ax,[.5 numel(method_labels)+.5]);
        set(ax,'TickLabelInterpreter','none','FontSize',8);
    end
    reference = 0;
    if initialization_study, reference = theta_true(p); end
    yline(ax, reference, 'k--', 'LineWidth', 1.0);
    grid(ax, 'on');
    box(ax, 'on');
    ylabel(ax, '$\hat{\theta}-\theta^\star$');
    if initialization_study, ylabel(ax,'Final parameter estimate'); end
    if p <= numel(param_labels)
        title(ax, sprintf('$%s$ final error', param_labels{p}));
        if initialization_study, title(ax,sprintf('$%s$',param_labels{p})); end
    else
        title(ax, sprintf('$\\theta_%d$ final error', p));
    end
    if ~initialization_study, ylim(ax, robust_axis_limits(values(:), 0)); end
end

for p = (n_params + 1):(n_rows * n_cols)
    axis(nexttile(tl), 'off');
end

title(tl, sprintf('%s Parametrization: Monte Carlo Final Parameter Errors', dataset.parametrization), 'FontWeight', 'bold', 'FontSize', 11);
if initialization_study
    title(tl,sprintf('%s: final estimates over %d initializations; fixed input/noise; dashed = truth', ...
        dataset.parametrization,size(errors,2)),'Interpreter','none');
end
end

function fig = plot_initialization_outcomes(dataset)
states = string(dataset.mc_summary.statuses);
names = cellstr_value(dataset.mc_summary.method_names);
labels = {'converged','stalled','max_iterations','failed / invalid','skipped'};
counts = [sum(states=="converged",2),sum(states=="stalled",2),sum(states=="max_iterations",2), ...
    sum(~ismember(states,["converged","stalled","max_iterations","ok","skipped"]),2),sum(states=="skipped",2)];
fig = figure('Color','w','Name',[dataset.parametrization ' initialization outcomes'], ...
    'Units','centimeters','Position',[3 3 26 14]);
ax = axes(fig); bar(ax,counts,'stacked');
xticks(ax,1:numel(names));
xticklabels(ax,cellfun(@method_figure_label,names,'UniformOutput',false)); xtickangle(ax,35);
ax.TickLabelInterpreter = 'none'; ylabel(ax,'Number of initializations');
legend(ax,labels,'Interpreter','none','Location','best'); grid(ax,'on');
title(ax,'Boxplots include finite outcomes; failed and skipped runs have no estimate','Interpreter','none');
end

function value = get_option(options,key,default)
if isfield(options,key), value=options.(key); else, value=default; end
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

function print_result_tables(datasets, mode)
if nargin < 2
    mode = '';
end

for i = 1:numel(datasets)
    if strcmpi(char_value(mode), 'MONTECARLO') && isfield(datasets(i), 'mc_summary') && ~isempty(datasets(i).mc_summary)
        print_montecarlo_result_table(datasets(i));
    else
        print_example_result_table(datasets(i));
    end
end
end

function print_example_result_table(dataset)
    fprintf('\n%s\n', repmat('=', 1, 108));
    fprintf('FINAL METHOD SUMMARY (%s parametrization)\n', dataset.parametrization);
    fprintf('%s\n', repmat('=', 1, 108));
    fprintf('%-18s %-12s %-14s %-14s %-12s %-12s\n', ...
        'Method', 'Status', 'Param err(%)', 'Output RMSE', 'Final d1', 'CPU time (s)');
    fprintf('%s\n', repmat('-', 1, 108));

    for m = 1:numel(dataset.results)
        result = dataset.results(m);
        theta_hat = double(result.theta_hat(:));
        final_d = NaN;
        if ~isempty(theta_hat)
            final_d = theta_hat(end);
        end
        fprintf('%-18s %-12s %-14.5f %-14.5f %-12.5f %-12.5f\n', ...
            display_method_label(result_name(result), dataset.parametrization), char_value(result.status), ...
            scalar_value(result.final_err), final_output_rmse(result), final_d, scalar_value(result.total_time));
    end
end

function print_montecarlo_result_table(dataset)
summary = dataset.mc_summary;
method_names = cellstr_value(summary.method_names);
n_methods = numel(method_names);

param_err_runs = orient_method_run_matrix(double(summary.final_errors), n_methods);
rmse_runs = final_values_from_mc_runs(summary.rmse_runs, n_methods);
[cpu_time_runs, cpu_source] = montecarlo_cpu_time_runs(summary, n_methods);

fprintf('\n%s\n', repmat('=', 1, 136));
fprintf('MONTE CARLO METHOD SUMMARY (%s parametrization)\n', dataset.parametrization);
fprintf('%s\n', repmat('=', 1, 136));
fprintf('%-22s %-12s %-6s %-14s %-14s %-14s %-14s %-14s %-14s\n', ...
    'Method', 'Status', 'N valid', 'Param mean', 'Param std', 'RMSE mean', 'RMSE std', 'CPU mean (s)', 'CPU std (s)');
fprintf('%s\n', repmat('-', 1, 136));

for m = 1:n_methods
    [param_mean, param_std, n_param] = mean_std_count(param_err_runs(m, :));
    [rmse_mean, rmse_std, n_rmse] = mean_std_count(rmse_runs(m, :));
    if isempty(cpu_time_runs)
        cpu_mean = NaN;
        cpu_std = NaN;
    else
        [cpu_mean, cpu_std] = mean_std_count(cpu_time_runs(m, :));
    end

    n_runs = min(n_param, n_rmse);
    fprintf('%-22s %-12s %-6d %-14.5f %-14.5f %-14.5f %-14.5f %-14.5f %-14.5f\n', ...
        display_method_label(method_names{m}, dataset.parametrization), ...
        result_status_for_method(dataset, method_names{m}), n_runs, ...
        param_mean, param_std, rmse_mean, rmse_std, cpu_mean, cpu_std);
end

if isempty(cpu_time_runs)
    fprintf(['\nNOTE: This saved Monte Carlo summary does not contain per-run CPU timing, ', ...
        'so CPU mean/std cannot be computed from this file.\n']);
elseif ~isempty(cpu_source)
    fprintf('\nTiming statistics include all recorded attempts from mc_summary.%s; N valid counts finite performance results.\n', cpu_source);
end
end

function matrix = orient_method_run_matrix(values, n_methods)
values = double(values);
if isempty(values)
    matrix = NaN(n_methods, 0);
    return;
end

if isvector(values)
    if n_methods == 1
        matrix = values(:).';
    elseif numel(values) == n_methods
        matrix = values(:);
    else
        matrix = reshape(values, n_methods, []);
    end
elseif size(values, 1) == n_methods
    matrix = values;
elseif size(values, 2) == n_methods
    matrix = values.';
else
    matrix = reshape(values, n_methods, []);
end
end

function final_values = final_values_from_mc_runs(runs, n_methods)
runs = double(runs);
if isempty(runs)
    final_values = NaN(n_methods, 0);
    return;
end

if ndims(runs) < 3
    final_values = orient_method_run_matrix(runs, n_methods);
    return;
end

if size(runs, 1) ~= n_methods && size(runs, 2) == n_methods
    runs = permute(runs, [2 1 3]);
end

n_runs = size(runs, 2);
final_values = NaN(n_methods, n_runs);
for m = 1:n_methods
    for r = 1:n_runs
        curve = squeeze(runs(m, r, :));
        curve = curve(isfinite(curve));
        if ~isempty(curve)
            final_values(m, r) = curve(end);
        end
    end
end
end

function [cpu_time_runs, source_field] = montecarlo_cpu_time_runs(summary, n_methods)
source_field = '';
candidate_fields = {'total_time_runs', 'cpu_time_runs', 'cpu_times', 'total_times'};
for i = 1:numel(candidate_fields)
    field_name = candidate_fields{i};
    if isfield(summary, field_name)
        cpu_time_runs = orient_method_run_matrix(summary.(field_name), n_methods);
        source_field = field_name;
        return;
    end
end

if isfield(summary, 'cum_time_runs')
    cpu_time_runs = final_values_from_mc_runs(summary.cum_time_runs, n_methods);
    source_field = 'cum_time_runs';
else
    cpu_time_runs = [];
end
end

function [mu, sigma, n] = mean_std_count(values)
values = double(values(:));
values = values(isfinite(values));
n = numel(values);
if n == 0
    mu = NaN;
    sigma = NaN;
    return;
end

mu = mean(values);
sigma = std(values, 0);
end

function status = result_status_for_method(dataset, method_name)
status = 'missing';
if ~isempty(dataset.mc_summary) && isfield(dataset.mc_summary,'statuses')
    names = cellstr_value(dataset.mc_summary.method_names);
    idx = find(strcmp(names,method_name),1);
    if ~isempty(idx)
        values = string(dataset.mc_summary.statuses(idx,:));
        types = unique(values);
        parts = arrayfun(@(t)sprintf('%s:%d',t,sum(values==t)),types,'UniformOutput',false);
        status = strjoin(parts,',');
    end
    return;
end
for i = 1:numel(dataset.results)
    if strcmp(result_name(dataset.results(i)), method_name)
        status = char_value(dataset.results(i).status);
        return;
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
% Diagnostics retain partial failed runs and methods absent from the first
% successful realization. Give every saved method a shared color as well.
for i = 1:numel(datasets)
    runs = datasets(i).all_runs;
    if isempty(runs), runs = {datasets(i).results}; end
    if ~iscell(runs), runs = {runs}; end
    for r = 1:numel(runs)
        results = runs{r};
        if iscell(results), results = [results{:}]; end
        for m = 1:numel(results)
            methods{end+1} = result_name(results(m)); %#ok<AGROW>
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

function [fig, fig_error] = plot_output_signals(dataset, method_names, colors, zoom_samples)
parametrization = dataset.parametrization;

if isempty(dataset.example_data) || ~all(isfield(dataset.example_data, {'r', 'nu', 'c'}))
    fig = unavailable_output_figure(parametrization, 'Output Signals', 'No saved output signal data available.');
    fig_error = unavailable_output_figure(parametrization, 'Output Errors', 'No saved output signal data available.');
    return;
end

r = double(dataset.example_data.r(:));
nu = double(dataset.example_data.nu(:));
c = double(dataset.example_data.c(:));
n_samples = min([zoom_samples, numel(r), numel(nu), numel(c)]);
if n_samples == 0
    fig = unavailable_output_figure(parametrization, 'Output Signals', 'Saved output signal data is empty.');
    fig_error = unavailable_output_figure(parametrization, 'Output Errors', 'Saved output signal data is empty.');
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
    fig_error = unavailable_output_figure(parametrization, 'Output Errors', ...
        'No plottable estimated output signals available.');
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

fig_error = figure('Name', sprintf('%s Output Errors', parametrization), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 24 9.5]);
ax_error = axes(fig_error);
hold(ax_error, 'on');
for idx = 1:n_estimates
    squared_error = (c(1:n_samples) - estimate_signals{idx}).^2;
    squared_error(squared_error == 0) = NaN; % Zero has no position on a log axis.
    plot(ax_error, x, squared_error, ...
        'Color', estimate_colors(idx, :), ...
        'DisplayName', latex_label(estimate_labels{idx}));
end
set(ax_error, 'YScale', 'log');
grid(ax_error, 'on');
box(ax_error, 'on');
xlim(ax_error, [1 n_samples]);
xlabel(ax_error, 'Sample $t$');
ylabel(ax_error, '$(c(t)-\hat{c}(t))^2$');
title(ax_error, sprintf('%s Parametrization: Squared Output Errors', parametrization));
legend(ax_error, 'Location', 'best', 'FontSize', 8);
end

function fig = unavailable_output_figure(parametrization, plot_name, message)
fig = figure('Name', sprintf('%s %s', parametrization, plot_name), ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 24 9.5]);
ax = axes(fig);
text(ax, 0.5, 0.5, message, ...
    'HorizontalAlignment', 'center', 'Units', 'normalized');
axis(ax, 'off');
end

function tf = is_ok_result(result)
tf = ismember(lower(strtrim(char_value(result.status))), {'ok','converged','stalled','max_iterations'});
end

function tf = is_plotted_result(result)
tf = is_ok_result(result) && result_cpu_time(result) > 0;
end

function tf = is_plottable_method(dataset, method_name)
tf = false;
if ~isempty(dataset.mc_summary) && isfield(dataset.mc_summary,'final_errors')
    names = cellstr_value(dataset.mc_summary.method_names);
    idx = find(strcmp(names,method_name),1);
    errors = orient_method_run_matrix(dataset.mc_summary.final_errors,numel(names));
    if ~isempty(idx), tf = any(isfinite(errors(idx,:))); end
    return;
end
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
label = method_figure_label(char_value(method_name));
label = strrep(label, 'RGLS', 'RLS');
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
