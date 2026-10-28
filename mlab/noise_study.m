function payload = noise_study(experiment_name, options)
% Innovation-standard-deviation sweep with paired records and a fixed start.
% noise_study('example_1',struct('study',struct('n_runs',1)))
if nargin < 1, experiment_name = 'example_CSTR'; end
if nargin < 2, options = struct(); end
payload = identification_study('noise',experiment_name,options);
end
