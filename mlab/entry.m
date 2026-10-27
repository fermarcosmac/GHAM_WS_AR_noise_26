%% MATLAB experiment and plotting entry point
% Run this setup once, then uncomment the desired block and run its section.
% Full settings: configs/<experiment_name>.json, configs/studies.json, and
% mlab/optim_configs.json. Standard calls overwrite their matching output files.
% Use the local-output examples for disposable runs. See README.md for details.

repo_root = pwd; % call from repo root!
addpath([repo_root '\mlab']);
experiment_name = 'example_1'; % 'example_1' or 'example_CSTR'
base_cfg = jsondecode(fileread(fullfile(repo_root,'configs',[experiment_name '.json'])));
results_dir = fullfile(repo_root,'results',experiment_name);
fig_dir = fullfile(repo_root,'figs');
export_figures = true; % false: display figures without exporting.
local_results_dir = fullfile(repo_root,'results','local');
local_fig_dir = fullfile(repo_root,'figs','local');
python_executable = 'python'; % Or the full path to your Python environment.

%% PL comparison: one input/noise realization, all configured optimizers
% cfg = base_cfg;
% cfg.mode = 'EXAMPLE';
% payload = comparison(cfg,struct('experiment_name',experiment_name, ...
%     'make_local_plots',true));

%% PL comparison: Monte Carlo input/noise realizations, fixed method starts
% cfg = base_cfg;
% cfg.mode = 'MONTECARLO';
% cfg.N_MC = 30; % Number of independent data records.
% payload = comparison(cfg,struct('experiment_name',experiment_name, ...
%     'make_local_plots',false));

%% Small PL comparison with explicit shared initialization and local output
% The supplied benchmarks have six parameters in [a, b, f2..., d] order.
% cfg = base_cfg;
% cfg.mode = 'EXAMPLE';
% cfg.selected_methods = {'WS-GGI','WS-GNI'};
% cfg.settings.lambda_g = 128;
% cfg.settings.K_max = 10; % Includes the initial state: at most nine updates.
% cfg.initialization = struct('policy','consistent','theta0',[0 0 1 0 0 0]);
% payload = comparison(cfg,struct('experiment_name',experiment_name, ...
%     'output_file',fullfile(local_results_dir,'comparison_pl_short.mat'), ...
%     'make_local_plots',false,'verbose',false));

%% PL initialization sensitivity: fixed data, random starts, all optimizers
% Uses initialization_study in configs/studies.json; progress is on by default.
% payload = initialization_study(experiment_name);

%% PL initialization sensitivity: override trial count/budget and save locally
% options = struct('experiment_cfg',base_cfg, ...
%     'study',struct('n_runs',5,'K_max',20,'progress_interval_seconds',10), ...
%     'output_file',fullfile(local_results_dir,'initialization_pl_short.mat'));
% payload = initialization_study(experiment_name,options);

%% PL record-length / finite-sample bias study: configured full experiment
% Load the record_length_study block explicitly (the JSON key includes 'length').
% FSM-only fields are omitted from the overrides; PL uses recursive filtering.
% n_runs independent data realizations, shared by all methods and nested lengths.
% theta0 is fixed. Bias error bars are mean error +/- 1.96*MCSE, not IQRs.
study_config_file = fullfile(repo_root,'configs','studies.json');
study_configs = jsondecode(fileread(study_config_file));
pl_record_cfg = study_configs.record_length_study;
fsm_fields = fieldnames(pl_record_cfg);
fsm_fields = fsm_fields(startsWith(fsm_fields,'fsm_'));
if ~isempty(fsm_fields), pl_record_cfg = rmfield(pl_record_cfg,fsm_fields); end
options = struct('experiment_cfg',base_cfg,'study_config',study_config_file, ...
    'study',pl_record_cfg);
payload = record_length_study(experiment_name,options);

%% PL record-length study: small local experiment
% options = struct('experiment_cfg',base_cfg, ...
%     'study',struct('n_runs',3,'K_max',10,'record_lengths',[256 512 1024]), ...
%     'output_file',fullfile(local_results_dir,'record_length_pl_short.mat'));
% payload = record_length_study(experiment_name,options);

%% Plot saved Monte Carlo comparisons AND available initialization studies
% Loads PL and FSM results for the selected benchmark. No identification runs.
% datasets = plot_comparison_results(struct('experiment_name',experiment_name, ...
%     'mode','MONTECARLO','fig_dir',fig_dir,'export_figures',export_figures));

%% Plot saved Monte Carlo comparisons only
% datasets = plot_comparison_results(struct('experiment_name',experiment_name, ...
%     'mode','MONTECARLO','include_initialization',false, ...
%     'fig_dir',fig_dir,'export_figures',export_figures));

%% Plot saved single-record comparisons only
% datasets = plot_comparison_results(struct('experiment_name',experiment_name, ...
%     'mode','EXAMPLE','include_initialization',false, ...
%     'fig_dir',fig_dir,'export_figures',export_figures));

%% Plot initialization studies only: final estimates, outcomes, conditioning
% Includes parameter-estimate boxplots with dashed true-parameter references.
datasets = plot_comparison_results(struct('study','initialization', ...
    'experiment_name',experiment_name,'fig_dir',fig_dir, ...
    'export_figures',export_figures));

%% Plot record-length studies only: signed bias, RMSE, outcome tables
% Reads record_length_pl.mat and record_length_fsm.mat when available.
plot_comparison_results(struct('study','record_length', ...
    'experiment_name',experiment_name,'fig_dir',fig_dir, ...
    'export_figures',export_figures));

%% Plot comparison conditioning only: one figure per parametrization
% datasets = plot_comparison_results(struct('experiment_name',experiment_name, ...
%     'mode','MONTECARLO','include_initialization',false,'diagnostics_only',true, ...
%     'fig_dir',fig_dir,'export_figures',export_figures));

%% Plot initialization conditioning only
% datasets = plot_comparison_results(struct('study','initialization', ...
%     'experiment_name',experiment_name,'diagnostics_only',true, ...
%     'fig_dir',fig_dir,'export_figures',export_figures));

%% Plot explicit files (PL, FSM, or custom/local outputs)
% The plotter detects comparison, initialization, or record-length payloads.
% data_files = {fullfile(local_results_dir,'initialization_pl_short.mat')};
% datasets = plot_comparison_results(struct('data_files',{data_files}, ...
%     'fig_dir',local_fig_dir,'export_figures',export_figures));

%% Direct record-length plot helper (change pl to fsm for Python results)
% [figures,outcomes] = plot_record_length_study(fullfile(results_dir,'record_length_pl.mat'), ...
%     struct('fig_dir',fig_dir,'export_figures',export_figures));

%% Direct conditioning helper and per-run summary table
% saved = load(fullfile(results_dir,'comparison_pl_montecarlo.mat'));
% [figures,report] = plot_optimization_diagnostics(saved,local_fig_dir, ...
%     [experiment_name '_montecarlo_pl'],export_figures);

%% PL noise-intensity sweep with matched seeds and local outputs
% cfg = base_cfg;
% cfg.mode = 'MONTECARLO';
% cfg.N_MC = 5;
% cfg.settings.lambda_g = 256;
% cfg.settings.K_max = 20;
% noise_levels = [0.01 0.05 0.10 0.20]; % Innovation standard deviations.
% for k = 1:numel(noise_levels)
%     cfg.settings.sigma_nu = noise_levels(k);
%     comparison(cfg,struct('experiment_name',experiment_name, ...
%         'make_local_plots',false,'verbose',false, ...
%         'output_file',fullfile(local_results_dir,sprintf('noise_pl_%02d.mat',k))));
% end

%% PL Newton Hessian-damping ablation with matched seeds and local outputs
% cfg = base_cfg;
% cfg.mode = 'MONTECARLO';
% cfg.N_MC = 3;
% cfg.selected_methods = {'WS-GNI'};
% cfg.settings.lambda_g = 128;
% cfg.settings.K_max = 10;
% cfg.initialization = struct('policy','consistent','theta0',[0 0 1 0 0 0]);
% optimizer_cfg = jsondecode(fileread(fullfile(entry_dir,base_cfg.matlab_optim_config)));
% damping = [0 1e-4 1e-2];
% for k = 1:numel(damping)
%     methods = optimizer_cfg.methods;
%     methods.WS_GNI.hessian_regularization = damping(k);
%     comparison(cfg,struct('experiment_name',experiment_name,'method_configs',methods, ...
%         'make_local_plots',false,'verbose',false, ...
%         'output_file',fullfile(local_results_dir,sprintf('gni_damping_pl_%02d.mat',k))));
% end

%% Short regression checks (temporary data; not manuscript-scale studies)
% test_comparison_diagnostics;
% test_identification_studies;
% Optional checks on actual Python study files:
% test_identification_studies(fullfile(results_dir,'initialization_fsm.mat'), ...
%     fullfile(results_dir,'record_length_fsm.mat'));

%% Original interactive PL example (WS-GGI)
% example.m selects its experiment internally and clears the workspace.
% Run the setup section again afterward before using other sections here.
% run(fullfile(entry_dir,'example.m'));

%% Python / FSM experiments launched from MATLAB
% Set python_executable above to an environment containing requirements.txt.
% The comparison script selects its benchmark inside main(), not via CLI.
% status = system(sprintf('"%s" "%s"',python_executable, ...
%     fullfile(repo_root,'python','comparison_torch.py')));
% status = system(sprintf('"%s" "%s" --experiment "%s"',python_executable, ...
%     fullfile(repo_root,'python','initialization_study.py'),experiment_name));
% status = system(sprintf('"%s" "%s" --experiment "%s"',python_executable, ...
%     fullfile(repo_root,'python','record_length_study.py'),experiment_name));
% Full CLI/API options are documented in README.md.
