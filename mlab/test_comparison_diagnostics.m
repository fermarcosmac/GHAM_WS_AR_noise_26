function test_comparison_diagnostics(fsm_file)
% Small numerical/export/plot checks. All test output uses a temporary folder.
if nargin < 1, fsm_file = ''; end
root = fileparts(mfilename('fullpath'));
folder = tempname(root);
assert(strcmp(fileparts(folder),root)); % Restrict recursive test cleanup to this folder.
mkdir(folder);
cleanup = onCleanup(@()cleanup_test(folder)); %#ok<NASGU>
visibility = get(groot,'defaultFigureVisible');
restore_visibility = onCleanup(@()set(groot,'defaultFigureVisible',visibility)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off');
metrics = matrix_diagnostics(diag([2 0]));
assert(isinf(metrics.condition) && metrics.rank==1 && metrics.negative_count==0);
metrics = matrix_diagnostics(diag([-2 1]));
assert(metrics.condition==2 && metrics.negative_count==1 && metrics.singular==0);

cfg = jsondecode(fileread(fullfile(root,'..','configs','example_1.json')));
cfg.settings.lambda_g = 32; cfg.settings.burn_in = 4; cfg.settings.K_max = 4;
cfg.settings.conv_threshold = 0; cfg.N_MC = 2;
cfg.initialization = struct('policy','consistent','theta0',[0 0 1 0 0 0]);
cfg.diagnostics = struct('enabled',true,'save_matrices',true);
options = struct('output_file',fullfile(folder,'pl.mat'),'make_local_plots',false);
p = comparison(cfg,options);
assert(p.schema_version==2 && numel(p.all_runs)==2);
assert(isequal(size(p.mc_summary.param_err_mean),[9 4]));
assert(all(p.mc_summary.param_err_count(6,:)==0)); % Unimplemented PL method.
assert(all(p.mc_summary.param_err_count(2,:)==2));
for m = 1:numel(p.results)
    r = p.results(m);
    assert(isequal(r.theta0,cfg.initialization.theta0(:)));
    assert(abs(r.total_wall_time-r.total_time-r.diagnostic_time)<1e-9);
    assert(strcmp(r.run_metadata.dataset_id,p.results(1).run_metadata.dataset_id));
end
gni = find(strcmp({p.results.name},'WS-GNI'));
d = p.results(gni).diagnostics;
assert(isequal(d.evaluation_iteration,(0:2).'));
assert(all(~isnan(d.raw_condition)) && all(d.raw_negative_count==0));
assert(isinf(d.raw_condition(1)) && all(isfinite(d.solver_condition)));
assert(~strcmp(p.all_runs{1}(gni).run_metadata.dataset_id,p.all_runs{2}(gni).run_metadata.dataset_id));
cfg.diagnostics.enabled = false;
off = comparison(cfg,options);
for m = 1:numel(p.results), assert(isequaln(p.results(m).theta_hist,off.results(m).theta_hist)); end
assert(~isfield(off.results(gni).diagnostics,'raw_condition'));

% Distinguish a zero step from satisfying the stationarity tolerance.
optim = jsondecode(fileread(fullfile(root,'optim_configs.json')));
options.method_configs = optim.methods;
options.method_configs.WS_GGI.step_scale = 0;
cfg.selected_methods = {'WS-GGI'}; cfg.mode = 'EXAMPLE';
cfg.settings.conv_threshold = 1e-8; cfg.settings.K_max = 5;
stalled = comparison(cfg,options);
assert(strcmp(stalled.results.status,'stalled'));
assert(isequal(stalled.results.active_mask,[true;true;false;false;false]));
cfg.stopping = struct('gradient_tolerance',1e100);
converged = comparison(cfg,options);
assert(strcmp(converged.results.status,'converged'));
options.method_configs.WS_GGI.step_scale = realmax;
cfg.mode = 'MONTECARLO';
failed = comparison(cfg,options);
assert(strcmp(failed.results.status,'invalid_update') && isnan(failed.results.final_err));
assert(failed.results.diagnostic_summary.rejected_count==1);
assert(all(isnan(failed.mc_summary.final_param_estimates),'all'));
assert(all(failed.mc_summary.param_err_count==0,'all'));
cfg.mode = 'EXAMPLE';
cfg.settings.K_max = 1;
initial = comparison(cfg,options);
assert(initial.results.accepted_updates==0 && initial.results.iterations==1);

% Exercise new and old files through the public plotting entry point.
save(options.output_file,'-struct','p','-v7');
plot_comparison_results(struct('data_files',{{options.output_file}}, ...
    'export_figures',false,'diagnostics_only',false));
close all;
[figures, report] = plot_optimization_diagnostics(p,folder,'test',true);
assert(numel(figures)==1 && height(report)==18);
assert(exist(fullfile(folder,'optimization_test.pdf'),'file')==2);
assert(numel(findobj(figures,'Type','axes'))==6);
all_axes = findobj(figures,'Type','axes');
assert(all(arrayfun(@(ax)isequal(ax.Color,[1 1 1]),all_axes)));
titles = arrayfun(@(ax)string(ax.Title.String),all_axes);
assert(any(titles=="Raw Hessian condition (finite samples)"));
assert(any(titles=="Solver matrix condition (finite samples)"));
assert(~any(contains(titles,"Fractions")));
for label = ["Raw Hessian condition (finite samples)","Solver matrix condition (finite samples)"]
    ax = all_axes(titles==label);
    curves = findall(ax,'Type','line');
    assert(any(arrayfun(@(curve)any(isfinite(curve.YData)),curves)));
end
objective_axes = all_axes(arrayfun(@(ax)strcmp(ax.Title.String,'Objective'),all_axes));
method_legend = objective_axes.Legend;
assert(numel(method_legend.String)==8);
assert(numel(unique(method_legend.String))==8);
assert(exist(fullfile(folder,'optimization_summary_test.csv'),'file')==2);
close all;
legacy = load(fullfile(root,'..','results','example_1','comparison_fsm_example.mat'));
[figures, report] = plot_optimization_diagnostics(legacy,folder,'legacy',false);
assert(isempty(figures) && isempty(report));
if ~isempty(fsm_file)
    fsm = load(fsm_file);
    [figures, report] = plot_optimization_diagnostics(fsm,folder,'fsm',false);
    assert(numel(figures)==1 && height(report)>0);
    close all;
end
fprintf('MATLAB comparison diagnostics checks passed.\n');
end

function cleanup_test(folder)
close all;
if exist(folder,'dir'), rmdir(folder,'s'); end
end
