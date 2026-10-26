function test_identification_studies(fsm_initialization_file, fsm_record_file)
% Small real-run, schema, and plot checks, with output confined to a temporary folder.
if nargin<1, fsm_initialization_file=''; end
if nargin<2, fsm_record_file=''; end
root=fileparts(mfilename('fullpath'));
results_root=char(java.io.File(fullfile(root,'..','results')).getCanonicalPath());
folder=tempname(results_root); mkdir(folder);
cleanup=onCleanup(@()cleanup_test(folder)); %#ok<NASGU>
visibility=get(groot,'defaultFigureVisible'); set(groot,'defaultFigureVisible','off');
restore=onCleanup(@()set(groot,'defaultFigureVisible',visibility)); %#ok<NASGU>
cfg=jsondecode(fileread(fullfile(root,'..','configs','example_1.json')));
cfg.settings.lambda_g=32; cfg.settings.burn_in=4; cfg.settings.K_max=3;
cfg.settings.conv_threshold=0; cfg.diagnostics.enabled=false;
options=struct('experiment_cfg',cfg,'study',struct('n_runs',3,'progress_interval_seconds',realmin), ...
    'output_file',fullfile(folder,'init.mat'));
console=evalc('p=initialization_study(''example_1'',options);');
assert(contains(console,'27/27 fits completed (100.0%)'));
assert(contains(console,'initialization 3/3') && contains(console,'skipped'));
assert(contains(console,'updates 1/2') && contains(console,'Finished 27/27 method fits'));
starts=p.study.initializations;
assert(size(unique(starts.','rows'),1)==3);
data_id=p.results(1).run_metadata.dataset_id;
for r=1:3
    for m=1:numel(p.all_runs{r})
        assert(isequal(p.all_runs{r}(m).theta0,starts(:,r)));
        assert(strcmp(p.all_runs{r}(m).run_metadata.dataset_id,data_id));
    end
end
options.study.progress=false;
quiet=evalc('again=initialization_study(''example_1'',options);');
assert(~contains(quiet,'[PL initialization]'));
assert(isequal(again.study.initializations,starts));
for r=1:3
    for m=1:numel(p.all_runs{r})
        assert(isequaln(p.all_runs{r}(m).theta_hist,again.all_runs{r}(m).theta_hist));
        assert(strcmp(p.all_runs{r}(m).run_metadata.run_id,again.all_runs{r}(m).run_metadata.run_id));
    end
end
% A default experiment selection must also discover saved initialization data.
% Exercise older payloads that carry INITIALIZATION mode but no study block.
legacy=rmfield(p,'study');
% Failure on realization one must not remove a method's diagnostic color.
legacy.results(1).status='failed'; legacy.results(1).final_err=NaN;
legacy.all_runs{1}(1).status='failed'; legacy.all_runs{1}(1).final_err=NaN;
legacy.mc_summary.statuses{1,1}='failed';
legacy.mc_summary.final_errors(1,1)=NaN;
legacy.mc_summary.final_param_estimates(1,1,:)=NaN;
legacy.mc_summary.final_param_errors(1,1,:)=NaN;
save(fullfile(folder,'initialization_pl.mat'),'-struct','legacy','-v7');
[~,experiment_tag]=fileparts(folder);
plot_options=struct('experiment_name',experiment_tag,'export_figures',false);
console=evalc('datasets=plot_comparison_results(plot_options);');
assert(numel(datasets)==1 && strcmp(datasets.study.type,'initialization'));
assert(contains(console,'Including initialization results:'));
figs=findall(groot,'Type','figure');
assert(any(contains(string({figs.Name}),'initialization outcomes')));
boxes=figs(contains(string({figs.Name}),'initialization final parameter estimates'));
assert(isscalar(boxes));
charts=findall(boxes,'Type','boxchart');
assert(numel(charts)==numel(p.experiment.theta_true));
first_title=['$' p.experiment.param_labels{1} '$'];
checked_first=false;
for chart=charts.'
    if strcmp(chart.Parent.Title.String,first_title)
        checked_first=true;
        expected=legacy.mc_summary.final_param_estimates(:,:,1);
        assert(isequal(sort(chart.YData(:)),sort(expected(isfinite(expected)))));
    end
end
assert(checked_first);
close all;

values=reshape([2 4 NaN],1,3,1);
stats=study_bias_statistics(values,{'converged','stalled','failed'},1,'finite');
assert(stats.bias==2 && abs(stats.mcse-1)<1e-12 && stats.count==2);
only=study_bias_statistics(values,{'converged','stalled','failed'},1,'converged');
assert(only.bias==1 && isnan(only.mcse));
options.study=struct('n_runs',2,'record_lengths',[16 24 32]);
options.output_file=fullfile(folder,'record.mat');
console=evalc('b=record_length_study(''example_1'',options);');
assert(contains(console,'54/54 fits completed (100.0%)'));
assert(contains(console,'N=32; realization 2/2') && contains(console,'ETA ~'));
assert(contains(console,'Finished 54/54 method fits'));
assert(isequal(size(b.study.final_param_estimates),[1 3 9 2 6]));
assert(isequal(size(b.study.bias),[1 3 9 6]));
assert(~strcmp(b.study.dataset_ids{1,1,1,1},b.study.dataset_ids{1,1,1,2}));
assert(isequal(b.study.theta0,[0;0;1;0;0;0]));
[figures,outcomes]=plot_record_length_study(b,struct('export_figures',true,'fig_dir',folder));
assert(numel(figures)==3 && height(outcomes)==27);
for ax=findall(figures,'Type','axes').'
    assert(strcmp(ax.Box,'on'));
end
assert(contains(figures(3).Name,'outcomes table'));
assert(isempty(findall(figures(3),'Type','legend')));
skip=strcmp(outcomes.MethodID,'WS-GGHAM-2-H');
assert(all(outcomes.Skipped(skip)==2 & outcomes.SkippedPct(skip)==100));
assert(all(outcomes.Failed(skip)==0 & outcomes.Finite(skip)==0));
assert(all(outcomes.Converged+outcomes.Stalled+outcomes.MaxIterations+ ...
    outcomes.Failed+outcomes.Skipped==outcomes.Trials));
assert(all(outcomes.Used==outcomes.Finite));
assert(exist(fullfile(folder,'record_length_example_1_pl.csv'),'file')==2);
outcome_csv=fullfile(folder,'record_length_outcomes_example_1_pl.csv');
assert(exist(outcome_csv,'file')==2);
exported=readtable(outcome_csv);
assert(height(exported)==height(outcomes) && isequal(exported.Skipped,outcomes.Skipped));
close all;
plot_comparison_results(struct('data_files',{{options.output_file}},'export_figures',false));
close all;
if ~isempty(fsm_initialization_file)
    plot_comparison_results(struct('data_files',{{fsm_initialization_file}},'export_figures',false));
    close all;
end
if ~isempty(fsm_record_file)
    figs=plot_record_length_study(fsm_record_file,struct('export_figures',false));
    assert(numel(figs)==6); close all;
end
fprintf('MATLAB identification study checks passed.\n');
end

function cleanup_test(folder)
close all;
root=fileparts(mfilename('fullpath'));
results_root=char(java.io.File(fullfile(root,'..','results')).getCanonicalPath());
assert(strcmp(fileparts(folder),results_root),'Unexpected cleanup path.');
if exist(folder,'dir'), rmdir(folder,'s'); end
end
