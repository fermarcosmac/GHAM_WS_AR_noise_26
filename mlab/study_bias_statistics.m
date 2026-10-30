function stats = study_bias_statistics(estimates, statuses, theta_true, population)
% Shapes: method x realization x parameter. Intervals are pointwise MC intervals.
assert(ismember(population,{'finite','converged'}),'summary_population must be finite or converged.');
M = size(estimates,1); n = numel(theta_true);
finite_runs = all(isfinite(estimates),3);
selected = finite_runs;
if strcmp(population,'converged')
    selected=selected & ismember(statuses,{'converged','sampling_complete'});
end
for p = 1:n
    slice = estimates(:,:,p); slice(~selected)=NaN; estimates(:,:,p)=slice;
end
errors = estimates-reshape(theta_true,1,1,n);
count = reshape(sum(isfinite(errors),2),M,n);
bias = reshape(mean(errors,2,'omitnan'),M,n);
variance = reshape(var(errors,0,2,'omitnan'),M,n); variance(count<2)=NaN;
mcse = sqrt(variance./max(count,1));
rmse = sqrt(reshape(mean(errors.^2,2,'omitnan'),M,n));
stats = struct('bias',bias,'variance',variance,'rmse',rmse,'mcse',mcse, ...
    'ci_low',bias-1.96*mcse,'ci_high',bias+1.96*mcse,'count',count, ...
    'finite_count',sum(finite_runs,2),'selected_count',sum(selected,2), ...
    'converged_count',sum(ismember(statuses,{'converged','sampling_complete'}),2), ...
    'skipped_count',sum(strcmp(statuses,'skipped'),2), ...
    'failed_count',sum(~ismember(statuses,{'converged','stalled','max_iterations','ok', ...
        'sampling_complete','sampling_warning','skipped'}),2));
end
