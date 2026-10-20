function payload = initialization_study(experiment_name, options)
% Fixed input/noise record, independent stable starts shared by all PL methods.
% initialization_study('example_1',struct('study',struct('n_runs',50)))
if nargin < 1, experiment_name = 'example_CSTR'; end
if nargin < 2, options = struct(); end
payload = identification_study('initialization',experiment_name,options);
end

%experimentName = 'example_1';
%options = readstruct('configs\studies.json');
%initialization_study(experimentName,options.initialization_study);