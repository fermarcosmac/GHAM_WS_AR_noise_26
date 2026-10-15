function summary = summarize_optimization_diagnostics(d)
summary = struct();
for prefix = {'raw', 'solver'}
    p = prefix{1};
    field = [p '_condition'];
    if ~isfield(d, field), continue; end
    values = d.(field);
    values = values(~isnan(values)); % Preserve Inf as evidence of singularity.
    if isempty(values), continue; end
    summary.([field '_initial']) = values(1);
    summary.([field '_final']) = values(end);
    summary.([field '_peak']) = max(values);
    summary.([p '_sample_count']) = numel(values);
    summary.([p '_singular_fraction']) = mean(isinf(values));
    negative = d.([p '_negative_count']);
    summary.([p '_negative_curvature_fraction']) = mean(negative(isfinite(negative)) > 0);
end
for key = {'solver_fallback', 'projected', 'rejected'}
    if isfield(d, key{1})
        summary.([key{1} '_count']) = sum(d.(key{1}), 'omitnan');
    else
        summary.([key{1} '_count']) = 0;
    end
end
end
