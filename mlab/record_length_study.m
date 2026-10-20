function payload = record_length_study(experiment_name, options)
% Signed parameter bias across nested record lengths and independent records.
% record_length_study('example_1',struct('study',struct('n_runs',100)))
if nargin < 1, experiment_name = 'example_CSTR'; end
if nargin < 2, options = struct(); end
payload = identification_study('record_length',experiment_name,options);
end
