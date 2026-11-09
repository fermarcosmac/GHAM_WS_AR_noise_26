function test_plot_noise_study()
% Check error aggregation, sample-SD bars, single-run display, and dispatch.
root=fileparts(fileparts(mfilename('fullpath')));
folder=tempname(fullfile(root,'results')); mkdir(folder);
cleanup=onCleanup(@()cleanup_test(folder,root)); %#ok<NASGU>
visibility=get(groot,'defaultFigureVisible'); set(groot,'defaultFigureVisible','off');
restore=onCleanup(@()set(groot,'defaultFigureVisible',visibility)); %#ok<NASGU>
values=NaN(3,3,2,2); states=repmat({'skipped'},3,3,2);
truth=[3;4];
for j=1:3
    values(j,1,1,:)=reshape(truth+[j;0],[1 1 1 2]);
    values(j,1,2,:)=reshape(truth-[j;0],[1 1 1 2]);
    values(j,2,1,:)=reshape(truth+[0;j],[1 1 1 2]);
    values(j,2,2,:)=reshape(truth+[0;100],[1 1 1 2]);
    states(j,1,:)={'converged','max_iterations'};
    states(j,2,:)={'converged','failed'};
end
study=struct('type','noise','sigma_nu_values',[0 .1 .5], ...
    'method_names',{{'WS-GGI','WS-GNI','RGLS'}},'param_labels',{{'a_1','b_1'}}, ...
    'n_runs',2,'theta_true',truth,'summary_population','finite', ...
    'final_param_estimates',values,'statuses',{states});
p=struct('parametrization','PL','experiment_name','test_noise_plot','study',study);
[figures,report]=plot_noise_study(p);
assert(numel(figures)==2 && height(report)==9);
rows=strcmp(report.Method,'WS-GGI');
assert(max(abs(report.MeanRelativeErrorPct(rows)-[20;40;60]))<1e-12);
assert(all(report.Used(rows)==2 & report.MCSE(rows)==0 & report.StdRelativeErrorPct(rows)==0));
rows=strcmp(report.Method,'WS-GNI');
assert(all(report.Used(rows)==1 & report.Failed(rows)==1 & ...
    isnan(report.MCSE(rows)) & isnan(report.StdRelativeErrorPct(rows))));
rows=strcmp(report.Method,'RGLS');
assert(all(report.Used(rows)==0 & report.Skipped(rows)==2 & isnan(report.MeanRelativeErrorPct(rows))));
close(figures);

% Ten unequal errors distinguish sample SD from MCSE and 95% mean intervals.
repeated_values=NaN(3,3,10,2);
repeated_states=repmat({'skipped'},3,3,10);
for j=1:3
    for r=1:10
        repeated_values(j,1,r,:)=reshape(truth+[j*r;0],[1 1 1 2]);
        repeated_states{j,1,r}='converged';
        repeated_values(j,2,r,:)=reshape(truth+[0;j*r],[1 1 1 2]);
        repeated_states{j,2,r}='converged';
    end
    repeated_states{j,2,10}='failed';
end
repeated=p;
repeated.study.n_runs=10;
repeated.study.final_param_estimates=repeated_values;
repeated.study.statuses=repeated_states;
for backend={'PL','FSM'}
    repeated.parametrization=backend{1};
    [figures,report]=plot_noise_study(repeated);
    rows=strcmp(report.Method,'WS-GGI');
    expected_mean=20*mean(1:10)*(1:3).';
    expected_std=20*std(1:10,0)*(1:3).';
    assert(max(abs(report.MeanRelativeErrorPct(rows)-expected_mean))<1e-12);
    assert(max(abs(report.StdRelativeErrorPct(rows)-expected_std))<1e-12);
    assert(max(abs(report.MCSE(rows)-expected_std/sqrt(10)))<1e-12);
    curve=findobj(figures(1),'Tag','noise_error','DisplayName','WS-GGI');
    bars=findall(curve.Parent,'Tag','noise_error_interval');
    assert(isscalar(bars));
    assert(max(abs(bars.YPositiveDelta(:)-expected_std))<1e-12);
    assert(max(abs(bars.YNegativeDelta(:)-expected_std))<1e-12);
    rows=strcmp(report.Method,'WS-GNI');
    assert(all(report.Used(rows)==9 & report.Failed(rows)==1));
    expected_std=20*std(1:9,0)*(1:3).';
    assert(max(abs(report.StdRelativeErrorPct(rows)-expected_std))<1e-12);
    close(figures);
end

% Singleton realization axes from MATLAB and SciPy must remain usable.
p.study.n_runs=1; p.study.final_param_estimates=values(:,:,1,:);
p.study.statuses=states(:,:,1);
[figures,report]=plot_noise_study(p,struct('export_figures',true,'fig_dir',folder));
assert(isempty(findall(figures,'Type','errorbar')));
assert(all(isnan(report.MCSE)));
assert(all(isnan(report.StdRelativeErrorPct)));
curve=findobj(figures(1),'Tag','noise_error','DisplayName','WS-GGI');
assert(isequal(curve.XData,[0 .1 .5]));
assert(max(abs(curve.YData-[20 40 60]))<1e-12);
assert(numel(findall(figures(2),'Tag','true_parameter'))==2);
for ax=findall(figures,'Type','axes').', assert(strcmp(ax.Box,'on')); end
assert(exist(fullfile(folder,'noise_error_test_noise_plot_pl.pdf'),'file')==2);
assert(exist(fullfile(folder,'noise_estimates_test_noise_plot_pl.pdf'),'file')==2);
exported=readtable(fullfile(folder,'noise_summary_test_noise_plot_pl.csv'));
assert(height(exported)==9 && isequal(exported.Used,report.Used));
assert(isequaln(exported.StdRelativeErrorPct,report.StdRelativeErrorPct));
close(figures);

% A single converged replicate has no MC interval, even in a repeated study.
p.study.n_runs=2; p.study.final_param_estimates=values; p.study.statuses=states;
p.study.summary_population='converged';
[figures,report]=plot_noise_study(p);
assert(all(report.Used(strcmp(report.Method,'WS-GGI'))==1));
assert(isempty(findall(figures,'Type','errorbar')));
close(figures);

% Same dispatcher handles PL, FSM, and explicit aggregate file paths.
p.study.n_runs=1; p.study.final_param_estimates=values(:,:,1,:); p.study.statuses=states(:,:,1);
path=fullfile(folder,'noise_pl.mat'); save(path,'-struct','p');
p.parametrization='FSM'; save(fullfile(folder,'noise_fsm.mat'),'-struct','p');
[~,experiment]=fileparts(folder);
plot_comparison_results(struct('study','noise','experiment_name',experiment,'export_figures',false));
assert(numel(findall(groot,'Type','figure'))==4); close all;
plot_comparison_results(struct('data_files',{{path}},'export_figures',false));
assert(numel(findall(groot,'Type','figure'))==2); close all;
fprintf('MATLAB noise plotting checks passed.\n');
end

function cleanup_test(folder,root)
close all;
results_root=char(java.io.File(fullfile(root,'results')).getCanonicalPath());
assert(strcmp(fileparts(folder),results_root),'Unexpected cleanup path.');
if exist(folder,'dir'), rmdir(folder,'s'); end
end
