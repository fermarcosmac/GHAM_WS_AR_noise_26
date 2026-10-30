function test_bayesian_comparison_plots()
% Focused BI_HMC export/plot/status checks without running a sampler.
root=fileparts(mfilename('fullpath'));
folder=tempname(root); mkdir(folder);
cleanup=onCleanup(@()cleanup_test(folder,root)); %#ok<NASGU>
visibility=get(groot,'defaultFigureVisible');
restore=onCleanup(@()set(groot,'defaultFigureVisible',visibility)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off'); rng(24);
truth=[-.2;1]; P=numel(truth);
base=struct('name','WS-GGI','status','converged','method_kind','optimization', ...
    'is_bayesian',false,'point_estimate_valid',true,'theta_hat',truth+.01, ...
    'theta_hist',[truth+.1 truth+.01],'param_err',[10 1],'rmse_hist',[.1 .01], ...
    'est_loss_hist',[1 .1],'cum_time',[0 .2],'total_time',.2,'total_wall_time',.3, ...
    'final_err',1,'final_output_rmse',.01,'diagnostic_time',.1, ...
    'accepted_updates',1,'run_metadata',struct('run_id','plot_fixture','dataset_id','same'), ...
    'diagnostics',struct(),'posterior_samples',[],'posterior_variance_samples',[]);
bayes=base; bayes.name='BI_HMC'; bayes.status='sampling_warning';
bayes.method_kind='bayesian'; bayes.is_bayesian=true;
bayes.theta_hist=zeros(P,0); bayes.param_err=[]; bayes.rmse_hist=[];
bayes.est_loss_hist=[]; bayes.cum_time=[]; bayes.total_time=2; bayes.total_wall_time=4;
bayes.final_err=.5; bayes.final_output_rmse=.005;
bayes.posterior_samples=reshape(repmat(truth.',10,1)+.01*randn(10,P),[2 5 P]);
bayes.posterior_variance_samples=.01+.001*rand(2,5);
bayes.diagnostics=struct('raw_condition',realmax); % Must not enter Hessian diagnostics.
experiment=struct('name','bayesian_plot_fixture','mode','EXAMPLE', ...
    'theta_true',truth,'sigma_nu',.1,'lambda_g',16,'burn_in',0, ...
    'seed',24,'dims',struct('na',1,'nb',1,'nf',1,'nd',0), ...
    'param_names',{{'a1','b1'}},'param_labels',{{'a_1','b_1'}});
payload=struct('parametrization','FSM','generated_at','test','experiment',experiment, ...
    'results',[base bayes],'all_runs',{{[base bayes]}}, ...
    'example_data',struct('r',sin((1:16).'),'nu',zeros(16,1),'c',zeros(16,1)));
file=fullfile(folder,'mixed.mat'); save(file,'-struct','payload');
datasets=plot_comparison_results(struct('data_files',{{file}},'export_figures',false, ...
    'normality',struct('enabled',false),'win_test',struct('enabled',false)));
assert(is_bayesian_result(datasets.results(2)));
figures=findall(groot,'Type','figure'); names=string({figures.Name});
posterior=figures(contains(names,'posterior distributions'));
assert(isscalar(posterior) && numel(findall(posterior,'Tag','bayesian_posterior'))==3);
runtime=figures(contains(names,'final accuracy and total runtime'));
markers=findall(runtime,'Tag','final_estimate_accuracy');
assert(numel(markers)==4);
bayes_markers=markers(contains(string({markers.DisplayName}),'BI'));
assert(all(arrayfun(@(line)isequal(line.XData,4),bayes_markers)));
for fig=figures(contains(names,'Parameter Trajectories') | contains(names,'Saved Comparison Metrics')).'
    curves=findall(fig,'Type','line');
    assert(~any(contains(string({curves.DisplayName}),'BI')));
end
[~,report]=plot_optimization_diagnostics(datasets,folder,'mixed',false);
assert(height(report)==1 && ~any(report.method=="BI_HMC"));
close all;

% With only Bayesian results, no optimizer trajectories are manufactured.
payload.results=bayes; payload.all_runs={bayes}; save(file,'-struct','payload');
dataset=plot_comparison_results(struct('data_files',{{file}},'export_figures',false));
figures=findall(groot,'Type','figure'); names=string({figures.Name});
assert(~any(contains(names,'Parameter Trajectories') | contains(names,'Saved Comparison Metrics')));
close all;
plot_bayesian_posteriors(dataset,struct('fig_dir',folder,'prefix','smoke','export_figures',true));
assert(exist(fullfile(folder,'posterior_smoke_BI_HMC.pdf'),'file')==2); close all;

% MATLAB omits a trailing singleton parameter dimension for P=1.
single=dataset; single.experiment.theta_true=1; single.experiment.param_labels={'b_1'};
single.results.theta_hat=1; single.results.posterior_samples=ones(2,5);
single.results.posterior_variance_samples=[];
figures=plot_bayesian_posteriors(single);
histograms=findall(figures,'Tag','bayesian_posterior');
assert(numel(histograms)==1 && numel(histograms.Data)==10); close all;

% Both finite sampler statuses enter paired comparisons; only passed enters
% the diagnostic-passing population used by the optional bias cohort.
states={'converged','stalled','max_iterations';'sampling_complete','sampling_warning','failed'};
summary=struct('method_names',{{'WS-GGI','BI_HMC'}},'final_errors',[2 3 4;1 2 NaN], ...
    'statuses',{states},'point_estimate_valid',[true true true;true true false]);
statdata=struct('parametrization','FSM','experiment',experiment,'mc_summary',summary);
statdata.experiment.mode='MONTECARLO';
[pairs,~,info]=pairwise_parameter_error_wins(statdata,struct('expected_runs',3));
assert(pairs.ValidPairs==2 && isequal(info.FiniteCounts,[3;2]));
values=repmat(reshape(truth,1,1,P),[2 3 1]); values(2,3,:)=NaN;
stats=study_bias_statistics(values,states,truth,'converged');
assert(isequal(stats.selected_count,[1;1]) && all(stats.failed_count==[0;1]));

study=struct('type','noise','sigma_nu_values',.1,'method_names',{{'WS-GGI','BI_HMC'}}, ...
    'param_labels',{{'a_1','b_1'}},'n_runs',3,'theta_true',truth,'summary_population','finite', ...
    'final_param_estimates',reshape(values,[1 2 3 P]),'statuses',{reshape(states,[1 2 3])}, ...
    'total_wall_time',reshape([.3 .4 .5;4 5 6],[1 2 3]));
[figures,report]=plot_noise_study(struct('parametrization','FSM','study',study));
assert(isequal(report.Used,[3;2]) && report.SamplerDiagnosticsPassed(2)==1 && report.SamplingWarning(2)==1);
assert(abs(report.MeanTotalWallTimeSec(2)-5)<1e-12 && all(report.TimingAttempts==3));
close(figures);
study.summary_population='converged';
[figures,report]=plot_noise_study(struct('parametrization','FSM','study',study));
assert(isequal(report.Used,[1;1])); close(figures);
fprintf('MATLAB BI_HMC comparison/posterior/status checks passed.\n');
end

function cleanup_test(folder,root)
close all;
folder=char(java.io.File(folder).getCanonicalPath());
root=char(java.io.File(root).getCanonicalPath());
assert(strcmp(fileparts(folder),root),'Unexpected cleanup path.');
if exist(folder,'dir'), rmdir(folder,'s'); end
end
