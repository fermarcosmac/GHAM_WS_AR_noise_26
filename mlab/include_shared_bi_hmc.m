function payload = include_shared_bi_hmc(payload, source_file, options)
% Add the saved Python BI_HMC reference to a PL presentation, in memory only.
% source_file identifies the PL file (its _fsm companion is discovered).
% options.bi_hmc_source can name an explicit Python MAT file;
% options.bi_hmc_resolution selects a saved FSM resolution (default: first).
% options.include_bi_hmc=false disables reuse. No MAT files are rewritten.
if nargin<2, source_file=''; end
if nargin<3, options=struct(); end
if ~opt(options,'include_bi_hmc',true) || ~strcmpi(string(payload.parametrization),'PL') ...
        || isfield(payload,'shared_bi_hmc'), return; end
source=opt(options,'bi_hmc_source','');
if isempty(source)
    if ~isempty(source_file)
        [folder,stem,ext]=fileparts(source_file);
        candidate=regexprep(stem,'_pl(?=_|$)','_fsm');
        if strcmp(candidate,stem), return; end
        source=fullfile(folder,[candidate ext]);
    else
        if isfield(payload,'experiment_name')
            name=char(string(payload.experiment_name));
            stem=[char(string(payload.study.type)) '_fsm.mat'];
        elseif isfield(payload,'experiment')
            name=char(string(payload.experiment.name));
            if isfield(payload,'study') && isfield(payload.study,'type') ...
                    && strcmp(string(payload.study.type),'initialization')
                stem='initialization_fsm.mat';
            else
                stem=['comparison_fsm_' lower(char(string(payload.experiment.mode))) '.mat'];
            end
        else, return;
        end
        source=fullfile(fileparts(fileparts(mfilename('fullpath'))),'results',name,stem);
    end
end
if isstruct(source), donor=source; source_name='explicit payload';
else
    source_name=char(string(source));
    if exist(source_name,'file')~=2, return; end
    donor=load(source_name);
end
assert(strcmpi(string(donor.parametrization),'FSM'), ...
    'include_shared_bi_hmc:Source','BI_HMC must come from Python/FSM saved results.');
aggregate=isfield(payload,'study') && isfield(payload.study,'type') && ...
    ismember(string(payload.study.type),["record_length","noise"]);
if aggregate
    names=string(donor.study.method_names(:));
else
    if ~isfield(donor,'results'), return; end
    names=string({donor.results.name});
end
index=find(names=="BI_HMC",1);
if isempty(index), return; end
if aggregate
    payload=merge_study(payload,donor,index,options);
else
    payload=merge_comparison(payload,donor,index);
end
payload.shared_bi_hmc=struct('source_file',source_name,'source_parametrization','FSM', ...
    'method','BI_HMC','reused_saved_results',true, ...
    'note','Python posterior estimates reused as a common reference; not a PL implementation.');
fprintf('PL presentation: reusing saved BI_HMC from %s\n',source_name);
end

function payload=merge_study(payload,donor,index,options)
t=payload.study; original=t; s=donor.study;
assert(strcmp(string(t.type),string(s.type)),'include_shared_bi_hmc:Mismatch','Study types differ.');
assert(strcmp(string(payload.experiment_name),string(donor.experiment_name)), ...
    'include_shared_bi_hmc:Mismatch','Benchmarks differ.');
same(t,s,'theta_true'); same(t,s,'param_names'); same(t,s,'theta0');
same_resolved_system(payload,donor);
record=strcmp(string(t.type),'record_length');
if record
    grid='record_lengths'; C=numel(t.resolution_labels); D=numel(s.resolution_labels);
else
    grid='sigma_nu_values'; C=1; D=1; same(t,s,'record_length');
end
[found,length_map]=ismember(double(t.(grid)(:)),double(s.(grid)(:)));
assert(all(found),'include_shared_bi_hmc:Mismatch','Python results lack requested study grid points.');
[found,run_map]=ismember(double(t.data_seeds(:)),double(s.data_seeds(:)));
assert(all(found) && numel(unique(run_map))==numel(run_map), ...
    'include_shared_bi_hmc:Mismatch','Python results lack requested data-seed repetitions.');
R=double(t.n_runs); S=double(s.n_runs); P=numel(t.theta_true);
assert(numel(run_map)==R,'include_shared_bi_hmc:Mismatch','Data seeds and repetitions differ.');
L=numel(t.(grid)); K=numel(s.(grid));
names=cellstr(string(t.method_names(:))); M=numel(names); Q=numel(s.method_names);
target=find(strcmp(names,'BI_HMC'),1);
if isempty(target), target=M+1; names{target}='BI_HMC'; end
U=numel(names);
resolution=opt(options,'bi_hmc_resolution',1);
if ischar(resolution) || isstring(resolution)
    resolution=find(string(s.resolution_labels)==string(resolution),1);
end
assert(isscalar(resolution) && resolution>=1 && resolution<=D && resolution==fix(resolution), ...
    'include_shared_bi_hmc:Resolution','Select a saved BI_HMC resolution by index or label.');
% All tensors use a temporary C x L x M x R x P layout, including noise (C=1).
run_fields={'final_param_estimates','statuses','dataset_ids','resolved_configs', ...
    'total_wall_time','total_wall_time_runs','total_time_runs','point_estimate_valid', ...
    'is_bayesian','sampler_diagnostics_passed','sampling_seeds','sampler_diagnostics_json'};
for k=1:numel(run_fields)
    field=run_fields{k};
    if ~isfield(s,field) && ~isfield(t,field), continue; end
    tail=1; if strcmp(field,'final_param_estimates'), tail=P; end
    a=study_array(original,field,[C L M R tail]);
    b=study_array(s,field,[D K Q S tail]);
    out=empty_like(a,[C L U R tail]); out(:,:,1:M,:,:)=a;
    out(:,:,target,:,:)=repmat(b(resolution,length_map,index,run_map,:),[C 1 1 1 1]);
    if record, t.(field)=out; else, t.(field)=reshape(out,[L U R tail]); end
end
t.method_names=reshape(names,1,[]);
% Recompute only the copied method's statistics under the PL cohort policy.
values=reshape(t.final_param_estimates,[C L U R P]);
states=reshape(string(t.statuses),[C L U R]);
for c=1:C
    for j=1:L
        v=reshape(values(c,j,target,:,:),[1 R P]);
        st=reshape(states(c,j,target,:),[1 R]);
        if isfield(t,'point_estimate_valid')
            valid=reshape(t.point_estimate_valid,[C L U R]);
            v(:,~reshape(valid(c,j,target,:),1,R),:)=NaN;
        end
        stats=study_bias_statistics(v,cellstr(st),t.theta_true,char(t.summary_population));
        stats.sampling_complete_count=sum(st=="sampling_complete");
        stats.sampling_warning_count=sum(st=="sampling_warning");
        for key=fieldnames(stats).'
            field=key{1}; tail=1;
            if ismember(field,{'bias','variance','rmse','mcse','ci_low','ci_high','count'}), tail=P; end
            a=study_array(t,field,[C L M tail]);
            % Previously expanded fields must retain the new method count.
            if isfield(t,field) && numel(t.(field))==C*L*U*tail
                out=reshape(t.(field),[C L U tail]);
            else
                out=empty_like(a,[C L U tail]); out(:,:,1:M,:)=a;
            end
            out(c,j,target,:)=reshape(stats.(field),[1 1 1 tail]);
            if record, t.(field)=out; else, t.(field)=reshape(out,[L U tail]); end
        end
    end
end
if record
    field='inverse_n_coefficients'; a=study_array(t,field,[C M P 2]);
    out=NaN(C,U,P,2); out(:,1:M,:,:)=a;
    bias=reshape(t.bias,[C L U P]);
    for c=1:C
        for p=1:P
            y=reshape(bias(c,:,target,p),[],1); keep=isfinite(y);
            if nnz(keep)>=3
                x=double(t.record_lengths(:));
                out(c,target,p,:)=reshape([ones(nnz(keep),1) 1./x(keep)]\y(keep),[1 1 1 2]);
            end
        end
    end
    t.(field)=out;
end
t.shared_bi_hmc_resolution=resolution;
t.shared_bi_hmc_note='Saved Python reference; seed numbers identify repetitions, not identical MATLAB/Python records.';
payload.study=t;
end

function a=study_array(s,field,shape)
if isfield(s,field) && ~isempty(s.(field))
    % The caller may have expanded statistics during a previous grid point.
    if numel(s.(field))==prod(shape), a=reshape(s.(field),shape); return; end
    if ismember(field,{'bias','variance','rmse','mcse','ci_low','ci_high','count', ...
            'finite_count','selected_count','converged_count','failed_count','skipped_count', ...
            'sampling_complete_count','sampling_warning_count'})
        expanded=shape; expanded(3)=numel(s.(field))/prod(shape([1 2 4]));
        a=reshape(s.(field),expanded); a=a(:,:,1:shape(3),:); return;
    end
    error('include_shared_bi_hmc:Shape','Unexpected saved shape for %s.',field);
end
if ismember(field,{'statuses','dataset_ids','resolved_configs','sampler_diagnostics_json'})
    a=repmat({''},shape);
elseif ismember(field,{'point_estimate_valid','is_bayesian','sampler_diagnostics_passed'})
    a=false(shape);
    if strcmp(field,'point_estimate_valid')
        P=numel(s.theta_true); v=reshape(s.final_param_estimates,[shape(1:4) P]);
        a=reshape(all(isfinite(v),5),shape);
    end
else, a=NaN(shape);
end
end

function out=empty_like(a,shape)
if iscell(a), out=repmat({''},shape);
elseif islogical(a), out=false(shape);
else, out=NaN(shape);
end
end

function payload=merge_comparison(payload,donor,index)
t=payload.experiment; s=donor.experiment;
for field={'name','mode','theta_true','param_names','lambda_g','sigma_nu','burn_in','seed'}
    same(t,s,field{1});
end
for field={'na','nb','nf','nd'}, same(t.dims,s.dims,field{1}); end
if isfield(payload,'study') && isfield(payload.study,'type')
    assert(isfield(donor,'study') && isfield(donor.study,'type') && ...
        strcmp(string(payload.study.type),string(donor.study.type)), ...
        'include_shared_bi_hmc:Mismatch','Comparison study types differ.');
    for field={'record_length','sigma_nu'}, same(payload.study,donor.study,field{1}); end
end
if isfield(payload,'mc_summary') && ~isempty(payload.mc_summary)
    assert(isfield(donor,'mc_summary') && ~isempty(donor.mc_summary), ...
        'include_shared_bi_hmc:Mismatch','Python Monte Carlo summary is missing.');
    payload.mc_summary=merge_summary(payload.mc_summary,donor.mc_summary,index);
end
result=donor.results(index);
result.shared_bi_hmc_reference=true;
if isfield(donor,'example_data'), result.presentation_example_data=donor.example_data; end
payload.results=replace_result(payload.results,result);
if isfield(payload,'all_runs') && ~isempty(payload.all_runs) && ...
        isfield(donor,'all_runs') && ~isempty(donor.all_runs)
    assert(numel(payload.all_runs)==numel(donor.all_runs), ...
        'include_shared_bi_hmc:Mismatch','Monte Carlo repetition counts differ.');
    for r=1:numel(payload.all_runs)
        source=donor.all_runs{r}; if iscell(source), source=[source{:}]; end
        k=find(string({source.name})=="BI_HMC",1);
        assert(~isempty(k),'include_shared_bi_hmc:MissingRun','Missing saved BI_HMC repetition.');
        result=source(k); result.shared_bi_hmc_reference=true;
        payload.all_runs{r}=replace_result(payload.all_runs{r},result);
    end
end
payload.experiment.selected_methods=cellstr(string({payload.results.name}));
end

function t=merge_summary(t,s,index)
names=cellstr(string(t.method_names(:))); M=numel(names); Q=numel(s.method_names);
R=size(t.final_errors,2); assert(size(s.final_errors,2)==R, ...
    'include_shared_bi_hmc:Mismatch','Monte Carlo repetition counts differ.');
target=find(strcmp(names,'BI_HMC'),1);
if isempty(target), target=M+1; names{target}='BI_HMC'; end
U=numel(names);
% Legacy MATLAB files have no validity matrix: preserve finite optimizer runs.
if ~isfield(t,'point_estimate_valid'), t.point_estimate_valid=isfinite(t.final_errors); end
if ~isfield(t,'statuses'), t.statuses=repmat({'unrecorded'},M,R); end
% Preserve legacy timings when the Python source supplies newer timing fields.
if ~isfield(t,'total_wall_time_runs') && isfield(t,'total_time_runs')
    t.total_wall_time_runs=t.total_time_runs;
end
fields=union(fieldnames(t),fieldnames(s));
for k=1:numel(fields)
    field=fields{k};
    if strcmp(field,'method_names'), continue; end
    if isfield(t,field), a=t.(field);
    else
        dims=size(s.(field)); dims(1)=M; a=empty_like(s.(field),dims);
    end
    assert(size(a,1)==M,'include_shared_bi_hmc:Shape','Summary %s must have one row per method.',field);
    dims=size(a); dims(1)=U; out=empty_like(a,dims);
    subs=repmat({':'},1,ndims(a)); subs{1}=1:M; out(subs{:})=a;
    if isfield(s,field)
        b=s.(field); assert(size(b,1)==Q,'Unexpected Python method axis.');
        subs{1}=target; donor_subs=repmat({':'},1,ndims(b)); donor_subs{1}=index;
        source=b(donor_subs{:});
        % HMC iteration arrays are all missing; match the PL optimizer budget.
        if numel(source)~=numel(out(subs{:}))
            if endsWith(field,'_count') && all(source==0,'all')
                out(subs{:})=0;
            else
                assert(all(isnan(double(source)),'all'),'Incompatible nonempty method history.');
            end
        else, out(subs{:})=source;
        end
    end
    t.(field)=out;
end
t.method_names=reshape(names,1,[]);
t.shared_reference_methods=reshape(strcmp(names,'BI_HMC'),1,[]);
end

function results=replace_result(results,result)
if iscell(results), results=[results{:}]; end
results=results(:).';
fields=union(fieldnames(results),fieldnames(result));
for k=1:numel(fields)
    field=fields{k};
    if ~isfield(results,field), [results.(field)]=deal([]); end
    if ~isfield(result,field), result.(field)=[]; end
end
result=orderfields(result,results);
target=find(string({results.name})=="BI_HMC",1);
if isempty(target), results(end+1)=result; else, results(target)=result; end
end

function same_resolved_system(a,b)
if ~isfield(a,'resolved_experiment_json') || ~isfield(b,'resolved_experiment_json'), return; end
x=jsondecode(a.resolved_experiment_json); y=jsondecode(b.resolved_experiment_json);
for field={'na','nb','nf','nd','theta_true'}, same(x.system,y.system,field{1}); end
same(x.settings,y.settings,'burn_in');
if ~strcmp(string(a.study.type),'noise'), same(x.settings,y.settings,'sigma_nu'); end
end

function same(a,b,field)
if ~isfield(a,field) || ~isfield(b,field), return; end
x=a.(field); y=b.(field);
if isnumeric(x) && isnumeric(y)
    match=numel(x)==numel(y) && all(abs(double(x(:))-double(y(:)))<=1e-12);
else, match=isequal(string(x(:)),string(y(:)));
end
assert(match,'include_shared_bi_hmc:Mismatch','Saved results disagree on %s.',field);
end

function value=opt(options,key,default)
if isfield(options,key), value=options.(key); else, value=default; end
end
