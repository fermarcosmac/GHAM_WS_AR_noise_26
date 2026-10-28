function test_noise_study()
% Real short fits: paired innovations, fixed starts, saved histories, and replay.
root = fileparts(fileparts(mfilename('fullpath')));
folder = tempname(fullfile(root,'results')); mkdir(folder);
cleanup = onCleanup(@()cleanup_test(folder,root)); %#ok<NASGU>
cfg = jsondecode(fileread(fullfile(root,'configs','example_1.json')));
cfg.settings.lambda_g = 32; cfg.settings.burn_in = 4;
cfg.settings.K_max = 3; cfg.settings.conv_threshold = 0; cfg.diagnostics.enabled = false;
cfg.selected_methods = {'RGLS','WS-GGI'};
options = struct('experiment_cfg',cfg,'study',struct('n_runs',2), ...
    'output_file',fullfile(folder,'noise.mat'));
console = evalc('p=noise_study(''example_1'',options);');
assert(contains(console,'20/20 fits completed (100.0%)'));
assert(contains(console,'sigma_nu=0.1; N=32; realization 2/2'));
assert(isequal(size(p.study.final_param_estimates),[5 2 2 6]));
assert(isequal(size(p.study.bias),[5 2 6]));
assert(numel(p.study.sigma_nu_values)==5 && p.study.sigma_nu_values(2)==0.1);
assert(isequal(p.study.innovation_variances,p.study.sigma_nu_values.^2));
assert(isequal(p.study.theta0,[0;0;1;0;0;0]));
for j = 1:5
    saved = load(p.study.case_files{j});
    sigma = p.study.sigma_nu_values(j);
    assert(saved.experiment.sigma_nu==sigma);
    resolved = jsondecode(saved.resolved_experiment_json);
    assert(resolved.settings.sigma_nu==sigma);
    unit_nu = saved.example_data.nu/sigma;
    if j==1
        baseline = saved.example_data; base_nu = unit_nu;
    end
    assert(isequal(saved.example_data.r,baseline.r));
    assert(max(abs(unit_nu-base_nu))<1e-12);
    % Remove the known recursively filtered noise, including its burn-in state.
    stream = RandStream('mt19937ar','Seed',p.study.data_seeds(1));
    randn(stream,36,1); nu_full = sigma*randn(stream,36,1);
    noise_full = filter(1,[1 cfg.system.theta_true(end)],nu_full);
    signal = saved.example_data.c-noise_full(5:end);
    if j==1, clean_signal = signal; end
    assert(max(abs(signal-clean_signal))<1e-12);
    for r = 1:2
        for m = 1:2
            result = saved.all_runs{r}(m);
            assert(isequal(result.theta0,p.study.theta0));
            assert(result.run_metadata.sigma_nu==sigma && result.run_metadata.noise_level_index==j);
            assert(result.run_metadata.data_seed==p.study.data_seeds(r));
            assert(strcmp(result.run_metadata.dataset_id,p.study.dataset_ids{j,m,r}));
        end
        assert(strcmp(p.study.dataset_ids{j,1,r},p.study.dataset_ids{j,2,r}));
    end
    assert(~strcmp(p.study.dataset_ids{j,1,1},p.study.dataset_ids{j,1,2}));
end
assert(numel(unique(p.study.dataset_ids(:,1,1)))==5);
options.study.progress = false; options.study.save_case_results = false;
options.output_file = fullfile(folder,'repeat.mat');
again = noise_study('example_1',options);
assert(isequaln(p.study.final_param_estimates,again.study.final_param_estimates));
assert(isempty(dir(fullfile(folder,'repeat_sigma*.mat'))));
assert(cfg.settings.sigma_nu==0.1); % Caller configuration is unchanged.
invalid = {[],[-0.1 0.1],[0.1 0.1],[0.2 0.1],NaN,Inf,[0.05 0.2;0.1 0.5]};
for j = 1:numel(invalid)
    options.study.sigma_nu_values = invalid{j}; caught = false;
    try
        noise_study('example_1',options);
    catch err
        caught = contains(err.message,'sigma_nu_values');
    end
    assert(caught,'Invalid noise levels must fail before fitting.');
end
options.study.sigma_nu_values = 0;
options.study.n_runs = 1;
zero = noise_study('example_1',options);
assert(zero.study.innovation_variances==0);
fprintf('MATLAB noise study checks passed.\n');
end

function cleanup_test(folder,root)
results_root = char(java.io.File(fullfile(root,'results')).getCanonicalPath());
assert(strcmp(fileparts(folder),results_root),'Unexpected cleanup path.');
if exist(folder,'dir'), rmdir(folder,'s'); end
end
