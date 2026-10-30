function tf = is_bayesian_result(result)
% Distinguish posterior sampling from optimization histories in saved results.
tf = false;
if isfield(result,'is_bayesian') && ~isempty(result.is_bayesian)
    tf = logical(result.is_bayesian(1));
end
if isfield(result,'method_kind') && ~isempty(result.method_kind)
    tf = tf || ismember(lower(strtrim(string(result.method_kind))), ...
        ["bayesian","bayesian_hmc","mcmc","hmc"]);
end
if isfield(result,'name')
    tf = tf || strcmpi(strtrim(string(result.name)),"BI_HMC");
end
end
