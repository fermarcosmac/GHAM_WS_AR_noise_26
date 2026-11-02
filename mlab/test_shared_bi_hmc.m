function test_shared_bi_hmc()
% Saved-result reuse: preserve optimizer rows, cohorts, timing, and provenance.
truth=[-.2;1]; P=2; R=3; L=3;
s=struct('type','record_length','method_names',{{'BI_HMC','WS-GGI'}}, ...
    'param_names',{{'a1','b1'}},'param_labels',{{'a_1','b_1'}}, ...
    'record_lengths',[8 16 32],'resolution_labels',{{'fsm_1','fsm_2'}}, ...
    'n_runs',R,'data_seeds',[10 11 12],'theta_true',truth,'theta0',[0;1], ...
    'summary_population','finite');
s.final_param_estimates=NaN(2,L,2,R,P);
s.statuses=repmat({'converged'},[2 L 2 R]);
for c=1:2
    for j=1:L
        for r=1:R
            s.final_param_estimates(c,j,1,r,:)=reshape(truth+[c;.01]*r/j,[1 1 1 1 P]);
            s.final_param_estimates(c,j,2,r,:)=reshape(truth+[.1;.1],[1 1 1 1 P]);
            s.statuses{c,j,1,r}='sampling_complete';
        end
        s.statuses{c,j,1,2}='sampling_warning';
        s.statuses{c,j,1,3}='failed'; s.final_param_estimates(c,j,1,3,:)=NaN;
    end
end
s.total_wall_time=reshape(1:2*L*2*R,[2 L 2 R]);
s.point_estimate_valid=all(isfinite(s.final_param_estimates),5);
source=struct('parametrization','FSM','experiment_name','shared_fixture','study',s);
t=s; t.resolution_labels={'recursive_PL'}; t.method_names={'WS-GGI'};
t.final_param_estimates=s.final_param_estimates(1,:,2,:,:);
t.statuses=s.statuses(1,:,2,:);
t=rmfield(t,{'total_wall_time','point_estimate_valid'});
for j=1:L
    stats=study_bias_statistics(reshape(t.final_param_estimates(1,j,:,:,:),[1 R P]), ...
        reshape(t.statuses(1,j,:,:),[1 R]),truth,'finite');
    for key=fieldnames(stats).'
        field=key{1}; tail=numel(stats.(field));
        t.(field)(1,j,:,:)=reshape(stats.(field),[1 1 1 tail]);
    end
end
target=struct('parametrization','PL','experiment_name','shared_fixture','study',t);
options=struct('bi_hmc_source',source);
merged=include_shared_bi_hmc(target,'',options);
assert(isequal(string(merged.study.method_names),["WS-GGI","BI_HMC"]));
assert(isequaln(merged.study.final_param_estimates(:,:,1,:,:),t.final_param_estimates));
assert(isequaln(merged.study.final_param_estimates(:,:,2,:,:),s.final_param_estimates(1,:,1,:,:)));
assert(all(merged.study.finite_count(:,:,2)==2));
assert(all(merged.study.selected_count(:,:,2)==2));
assert(all(merged.study.converged_count(:,:,2)==1));
assert(all(merged.study.failed_count(:,:,2)==1));
assert(all(isnan(merged.study.total_wall_time(:,:,1,:)),'all'));
assert(isequal(merged.study.total_wall_time(:,:,2,:),s.total_wall_time(1,:,1,:)));
assert(isequaln(include_shared_bi_hmc(merged,'',options),merged));
options.include_bi_hmc=false;
assert(isequaln(include_shared_bi_hmc(target,'',options),target));
options.include_bi_hmc=true; options.bi_hmc_resolution='fsm_2';
selected=include_shared_bi_hmc(target,'',options);
assert(isequaln(selected.study.final_param_estimates(:,:,2,:,:),s.final_param_estimates(2,:,1,:,:)));
target.study.summary_population='converged';
selected=include_shared_bi_hmc(target,'',options);
assert(all(selected.study.selected_count(:,:,2)==1));
% Align a subset of lengths and a reordered repetition list by saved keys.
subset=target; subset.study.record_lengths=[8 32]; subset.study.data_seeds=[12 10 11];
for key={'final_param_estimates','statuses'}
    field=key{1}; subset.study.(field)=target.study.(field)(:,[1 3],:,[3 1 2],:);
end
for key={'bias','variance','rmse','mcse','ci_low','ci_high','count','finite_count', ...
        'selected_count','converged_count','skipped_count','failed_count'}
    field=key{1}; subset.study.(field)=target.study.(field)(:,[1 3],:,:);
end
selected=include_shared_bi_hmc(subset,'',options);
assert(isequaln(selected.study.final_param_estimates(:,:,2,:,:),s.final_param_estimates(2,[1 3],1,[3 1 2],:)));
bad=target; bad.study.theta_true=[0;1];
try
    include_shared_bi_hmc(bad,'',options); error('test_shared_bi_hmc:NoError','Expected a mismatch.');
catch exc, assert(strcmp(exc.identifier,'include_shared_bi_hmc:Mismatch')); end
% Noise tensors have no resolution axis; singleton runs and parameters work.
noise=struct('type','noise','method_names',{{'WS-GGI'}},'sigma_nu_values',[.1 .2], ...
    'record_length',16,'n_runs',1,'data_seeds',10,'theta_true',1, ...
    'param_names',{{'b1'}},'theta0',1,'summary_population','finite', ...
    'final_param_estimates',reshape([1.1 1.2],[2 1 1]), ...
    'statuses',{{'converged';'converged'}});
ns=noise; ns.method_names={'BI_HMC'}; ns.final_param_estimates=reshape([.9 .8],[2 1 1]);
ns.statuses={'sampling_complete';'sampling_warning'}; ns.total_wall_time=[2;3];
nt=struct('parametrization','PL','experiment_name','shared_noise_fixture','study',noise);
nd=nt; nd.parametrization='FSM'; nd.study=ns;
nm=include_shared_bi_hmc(nt,'',struct('bi_hmc_source',nd));
assert(isequal(reshape(nm.study.final_param_estimates,[2 2]),[1.1 .9;1.2 .8]));
assert(all(abs(nm.study.bias(:,2)-[-.1;-.2])<1e-12));
% Replacing an existing skipped entry does not append a duplicate method.
placeholder=merged; placeholder=rmfield(placeholder,'shared_bi_hmc');
placeholder.study.statuses(:,:,2,:)={'skipped'};
replaced=include_shared_bi_hmc(placeholder,'',struct('bi_hmc_source',source));
assert(numel(replaced.study.method_names)==2);
assert(isequaln(replaced.study.final_param_estimates,merged.study.final_param_estimates));
test_comparison_reuse();
fprintf('MATLAB shared BI_HMC reuse/alignment/cohort/hypothesis checks passed.\n');
end

function test_comparison_reuse()
base=struct('name','WS-GGI','status','converged','theta_hat',1,'theta0',1, ...
    'run_metadata',struct('dataset_id','pl_record','mc_index',1));
bayes=base; bayes.name='BI_HMC'; bayes.status='sampling_complete';
bayes.run_metadata.dataset_id='python_record'; bayes.posterior_samples=ones(2,4);
e=struct('name','shared_comparison_fixture','mode','MONTECARLO','theta_true',1, ...
    'lambda_g',8,'sigma_nu',.1,'seed',1,'burn_in',0, ...
    'dims',struct('na',0,'nb',1,'nf',1,'nd',0));
t=struct('parametrization','PL','experiment',e,'results',base);
s=t; s.parametrization='FSM'; s.results=bayes;
t.mc_summary=struct('method_names',{{'WS-GGI'}},'final_errors',[1 2 3], ...
    'total_time_runs',[.1 .2 .3]);
s.mc_summary=struct('method_names',{{'BI_HMC'}},'final_errors',[.1 .2 NaN], ...
    'statuses',{{'sampling_complete','sampling_warning','failed'}}, ...
    'total_time_runs',[1 2 3],'total_wall_time_runs',[2 3 4], ...
    'point_estimate_valid',[true true false]);
t.all_runs=cell(1,3); s.all_runs=cell(1,3);
for r=1:3
    b=base; b.run_metadata.dataset_id=['pl_' num2str(r)]; b.run_metadata.mc_index=r;
    d=bayes; d.run_metadata.dataset_id=['python_' num2str(r)]; d.run_metadata.mc_index=r;
    t.all_runs{r}=b; s.all_runs{r}=d;
end
merged=include_shared_bi_hmc(t,'',struct('bi_hmc_source',s));
assert(isequal(merged.mc_summary.total_wall_time_runs,[.1 .2 .3;2 3 4]));
assert(isequal(merged.mc_summary.point_estimate_valid,[true true true;true true false]));
assert(numel(merged.all_runs{1})==2 && ~isempty(merged.results(2).posterior_samples));
[pairs,trials]=pairwise_parameter_error_wins(merged,struct('expected_runs',3));
assert(pairs.Test=="independent_rank_sum" && isnan(pairs.ValidPairs));
assert(abs(pairs.RawP-ranksum([1 2 3],[.1 .2]))<1e-12);
assert(pairs.ValidA==3 && pairs.ValidB==2 && isnan(pairs.CILower));
assert(all(trials.Outcome=="unpaired_sample_columns"));
end
