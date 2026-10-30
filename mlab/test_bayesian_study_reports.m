function test_bayesian_study_reports()
% Check mixed optimizer/sampler record-length cohorts and total-time reports.
root=fileparts(mfilename('fullpath')); folder=tempname(root); mkdir(folder);
cleanup=onCleanup(@()cleanup_test(folder,root)); %#ok<NASGU>
visibility=get(groot,'defaultFigureVisible');
restore=onCleanup(@()set(groot,'defaultFigureVisible',visibility)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off');
truth=[-.2;1]; M=2; R=3; P=2; L=2;
values=NaN(1,L,M,R,P); statuses=cell(1,L,M,R);
study=struct('type','record_length','method_names',{{'WS-GGI','BI_HMC'}}, ...
    'param_labels',{{'a_1','b_1'}},'resolution_labels',{{'test'}}, ...
    'record_lengths',[8 16],'n_runs',R,'theta_true',truth,'summary_population','finite');
for j=1:L
    states={'converged','stalled','max_iterations';'sampling_complete','sampling_warning','failed'};
    estimates=repmat(reshape(truth,1,1,P),[M R 1]);
    for m=1:M
        for r=1:R, estimates(m,r,:)=reshape(truth+[.02;.01]*r/(m*j),[1 1 P]); end
    end
    estimates(2,3,:)=NaN;
    values(1,j,:,:,:)=reshape(estimates,[1 1 M R P]);
    statuses(1,j,:,:)=reshape(states,[1 1 M R]);
    stats=study_bias_statistics(estimates,states,truth,'finite');
    for field={'bias','mcse','rmse','variance','count'}
        study.(field{1})(1,j,:,:)=reshape(stats.(field{1}),[1 1 M P]);
    end
    for field={'finite_count','converged_count','failed_count','skipped_count','selected_count'}
        study.(field{1})(1,j,:)=reshape(stats.(field{1}),[1 1 M]);
    end
    study.total_wall_time(1,j,:,:)=reshape([.3 .4 .5;4 5 6]*j,[1 1 M R]);
end
study.final_param_estimates=values; study.statuses=statuses;
payload=struct('parametrization','FSM','experiment_name','bayesian_study_fixture','study',study);
[figures,report,aggregate]=plot_record_length_study(payload, ...
    struct('fig_dir',folder,'export_figures',true));
assert(numel(figures)==4 && height(report)==L*M);
sampler=strcmp(report.MethodID,'BI_HMC');
assert(all(report.SamplingWarning(sampler)==1 & report.Converged(sampler)==1));
assert(all(report.Failed(sampler)==1 & report.Used(sampler)==2));
assert(all(abs(report.MeanTotalWallTimeSec(sampler)-[5;10])<1e-12));
assert(all(report.TimingAttempts==R));
assert(all(report.Converged+report.Stalled+report.MaxIterations+report.SamplingWarning+ ...
    report.Failed+report.Skipped==report.Trials));
csv=readtable(fullfile(folder,'record_length_outcomes_bayesian_study_fixture_fsm.csv'));
assert(isequal(csv.SamplingWarning,report.SamplingWarning));
% CSV text round-tripping can change the last binary floating-point bit.
assert(all(abs(csv.MeanTotalWallTimeSec-report.MeanTotalWallTimeSec)<1e-12));
assert(all(aggregate.Used==[repmat(3,L,1);repmat(2,L,1)]));
close(figures);
payload.study.summary_population='converged';
[figures,aggregate]=plot_record_length_aggregate_bias(payload,struct('export_figures',false));
assert(all(aggregate.Used==1)); close(figures);
fprintf('MATLAB BI_HMC study outcome/cohort/runtime checks passed.\n');
end

function cleanup_test(folder,root)
close all;
folder=char(java.io.File(folder).getCanonicalPath());
root=char(java.io.File(root).getCanonicalPath());
assert(strcmp(fileparts(folder),root),'Unexpected cleanup path.');
if exist(folder,'dir'), rmdir(folder,'s'); end
end
