function [report, info] = parameter_error_normality(dataset, options)
% Shapiro-Wilk tests on raw final relative parameter errors, one per method.
% R2026a has no native swtest: use scipy.stats.shapiro through a JSON bridge.
% Configure options.python_executable if Python/SciPy are not on PATH.
% Holm adjustment is a separate family from the optimizer win tests.
if nargin<2, options=struct(); end
assert(strcmpi(string(dataset.experiment.mode),'MONTECARLO'), ...
    'Normality diagnostics here are restricted to Monte Carlo realizations.');
alpha=opt(options,'alpha',0.05);
assert(isscalar(alpha) && alpha>0 && alpha<1,'alpha must lie in (0,1).');
names=string(dataset.mc_summary.method_names(:)); M=numel(names);
errors=double(dataset.mc_summary.final_errors);
if size(errors,1)~=M, errors=errors.'; end
assert(size(errors,1)==M,'Expected one error row per method.');
assert(~any(errors(isfinite(errors))<0),'Parameter errors must be nonnegative.');
R=size(errors,2);
expected_runs=opt(options,'expected_runs',30);
if ~isempty(expected_runs) && R~=expected_runs
    warning('parameter_error_normality:RunCount','Expected %d trials; found %d.',expected_runs,R);
end
states=repmat("unrecorded",size(errors));
if isfield(dataset.mc_summary,'statuses')
    states=string(dataset.mc_summary.statuses);
    if size(states,1)~=M, states=states.'; end
    assert(isequal(size(states),size(errors)),'Statuses and errors must align.');
end
valid=isfinite(errors) & ismember(states,["ok","converged","stalled","max_iterations", ...
    "sampling_complete","sampling_warning","unrecorded"]);
if isfield(dataset.mc_summary,'point_estimate_valid')
    recorded_valid=logical(dataset.mc_summary.point_estimate_valid);
    if size(recorded_valid,1)~=M, recorded_valid=recorded_valid.'; end
    assert(isequal(size(recorded_valid),size(valid)),'Point-estimate validity and errors must align.');
    valid=valid & recorded_valid;
end
samples=cell(M,1); sample_mean=NaN(M,1); sample_std=NaN(M,1);
for m=1:M
    samples{m}=errors(m,valid(m,:));
    if ~isempty(samples{m})
        sample_mean(m)=mean(samples{m}); sample_std(m)=std(samples{m},0);
    end
end
python_executable=char(opt(options,'python_executable','python'));
root=fileparts(fileparts(mfilename('fullpath')));
backend=fullfile(root,'python','shapiro_wilk_samples.py');
input_file=[tempname '.json']; output_file=[tempname '.json'];
cleanup=onCleanup(@()remove_temp_files(input_file,output_file)); %#ok<NASGU>
fid=fopen(input_file,'w'); assert(fid~=-1,'Cannot create normality test input.');
fprintf(fid,'%s',jsonencode(struct('samples',{samples}))); fclose(fid);
command=sprintf('%s %s --input %s --output %s', ...
    quoted(python_executable),quoted(backend),quoted(input_file),quoted(output_file));
[status,message]=system(command);
assert(status==0,'parameter_error_normality:Python', ...
    'Shapiro-Wilk backend failed. Configure normality.python_executable with a Python/SciPy environment.\n%s',message);
result=jsondecode(fileread(output_file));
W=double(result.W(:)); p=double(result.P(:)); state=string(result.Status(:));
assert(numel(W)==M && numel(p)==M,'Shapiro-Wilk backend returned the wrong method count.');
adjusted=NaN(M,1); tested=isfinite(p);
if any(tested)
    indices=find(tested); [sorted,order]=sort(p(tested));
    adjusted(indices(order))=min(1,cummax((sum(tested):-1:1)'.*sorted));
end
decision=state;
decision(tested & adjusted<alpha)="Reject normality";
decision(tested & adjusted>=alpha)="Normality not rejected";
decision(sum(valid,2)==0)="No valid observations";
labels=names;
if isfield(options,'method_labels'), labels=string(options.method_labels(:)); end
assert(numel(labels)==M,'method_labels must match methods.');
report=table(repmat(string(dataset.experiment.name),M,1),repmat(string(dataset.parametrization),M,1), ...
    names,labels,repmat(R,M,1),sum(valid,2),R-sum(valid,2),sample_mean,sample_std,W,p,adjusted, ...
    tested & p<alpha,tested & adjusted<alpha,decision, ...
    'VariableNames',{'Benchmark','Parametrization','Optimizer','Label','TotalRuns','ValidSamples', ...
    'ExcludedSamples','MeanError','StdError','W','RawP','AdjustedP','RejectRaw','RejectHolm','Decision'});
info=struct('Study','montecarlo','Metric','100*norm(theta_hat-theta_true,2)/norm(theta_true,2)', ...
    'Alpha',alpha,'Adjustment','Holm within benchmark/parametrization across testable optimizers', ...
    'FamilySize',sum(tested),'Engine',string(result.Engine),'EngineVersion',string(result.Version), ...
    'Interpretation','Non-rejection does not establish normality; all finite extreme values retained');
if isfield(dataset,'file'), info.SourceFile=string(dataset.file); end
end

function value=opt(options,key,default)
if isfield(options,key), value=options.(key); else, value=default; end
end

function value=quoted(value)
assert(~any(ismember(value,[char(34) char(10) char(13) '%'])), ...
    'Executable/file paths must not contain quotes, newlines, or percent signs.');
value=['"' value '"'];
end

function remove_temp_files(varargin)
for k=1:nargin
    if exist(varargin{k},'file')==2, delete(varargin{k}); end
end
end
