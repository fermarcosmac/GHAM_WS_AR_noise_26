function payload = identification_study(kind, experiment_name, options)
% Shared orchestration; optimization updates remain in comparison.m.
root = fileparts(fileparts(mfilename('fullpath')));
cfg = jsondecode(fileread(fullfile(root,'configs',[experiment_name '.json'])));
cfg = field_or(options,'experiment_cfg',cfg);
config_path = field_or(options,'study_config',fullfile(root,'configs','studies.json'));
all_settings = jsondecode(fileread(config_path));
study = all_settings.([kind '_study']);
overrides = field_or(options,'study',struct());
for key = fieldnames(overrides).', study.(key{1}) = overrides.(key{1}); end
R = study.n_runs;
assert(isscalar(R) && R>=1 && R==fix(R),'n_runs must be a positive integer.');
cfg.N_MC = R; cfg.make_local_plots = false;
cfg.settings.enforce_fixed_iterations = false;
if isfield(study,'K_max'), cfg.settings.K_max = study.K_max; end
sys = cfg.system;
n = sys.na+sys.nb+sys.nf-1+field_or(sys,'nd',0);
center = zeros(n,1); if sys.nb>0, center(sys.na+1)=1; end
output = field_or(options,'output_file',fullfile(root,'results',experiment_name,[kind '_pl.mat']));
folder = fileparts(output); if ~isempty(folder) && ~exist(folder,'dir'), mkdir(folder); end
base_options = struct('make_local_plots',false,'verbose',field_or(options,'verbose',false), ...
    'experiment_name',experiment_name,'output_file',output);
if isfield(options,'method_configs'), base_options.method_configs = options.method_configs; end

switch kind
    case 'initialization'
        progress = study_progress('PL initialization',cfg.selected_methods,R,cfg.settings.lambda_g,1,study);
        progress.stage('Preparing stable starts and the fixed data record');
        if progress.enabled, base_options.progress_callback = progress.callback; end
        starts = draw_starts(study,sys,center);
        cfg.mode = 'INITIALIZATION';
        cfg.seed = field_or(study,'data_seed',field_or(cfg,'seed',42));
        base_options.initializations = starts;
        payload = comparison(cfg,base_options);
        study.type = 'initialization'; study.randomized_variable = 'parameter_initialization';
        study.data_seed = cfg.seed; study.initializations = starts;
        payload.study = study;
        for r = 1:R
            for m = 1:numel(payload.all_runs{r})
                payload.all_runs{r}(m).run_metadata.initialization_index = r;
                payload.all_runs{r}(m).run_metadata.initialization_seed = study.initialization_seed;
            end
        end
        payload.results = payload.all_runs{1};
    case 'noise'
        levels = study.sigma_nu_values(:).';
        assert(isvector(study.sigma_nu_values) && ~isempty(levels) && ...
            all(isfinite(levels)) && all(levels>=0) && all(diff(levels)>0), ...
            'sigma_nu_values must be nonempty, finite, nonnegative, unique, and increasing.');
        theta0 = parameter_vector(study.theta0,n,center);
        assert(is_stable(theta0,sys,1),'theta0 denominators must be stable.');
        assert(ismember(study.summary_population,{'finite','converged'}), ...
            'summary_population must be finite or converged.');
        cfg.initialization = struct('policy','consistent','theta0',theta0);
        cfg.mode = 'MONTECARLO';
        cfg.seed = field_or(study,'data_seed',field_or(cfg,'seed',42));
        study.type = 'noise'; study.varied_variable = 'innovation_standard_deviation';
        study.data_seed = cfg.seed; study.theta0 = theta0;
        methods = cellstr(string(cfg.selected_methods)); methods = methods(:).';
        M = numel(methods); L = numel(levels); length = cfg.settings.lambda_g;
        progress = study_progress('PL noise',methods,R,repmat(length,1,L),1,study);
        estimates = NaN(L,M,R,n); statuses = cell(L,M,R);
        resolved_configs = cell(L,M,R); dataset_ids = cell(L,M,R);
        case_files = repmat({''},1,L); statistics = struct();
        for j = 1:L
            cfg.settings.sigma_nu = levels(j);
            progress.stage(sprintf('Noise level %d/%d: sigma_nu=%g',j,L,levels(j)));
            data_runs = cell(R,1);
            for r = 1:R
                % Reset the stream at every level, including pre-burn-in noise.
                % All methods share the input and scaled innovations for this trial.
                data_runs{r} = reference_data(cfg,length,cfg.seed+r-1);
            end
            run_options = base_options;
            run_options.data_runs = data_runs; run_options.save_results = false;
            if progress.enabled
                run_options.progress_callback = @(event) noise_progress(progress,event,levels(j));
            end
            run = comparison(cfg,run_options);
            for r = 1:R
                for m = 1:M
                    run.all_runs{r}(m).run_metadata.study_type = 'noise';
                    run.all_runs{r}(m).run_metadata.noise_level_index = j;
                    run.all_runs{r}(m).run_metadata.sigma_nu = levels(j);
                    resolved_configs{j,m,r} = run.all_runs{r}(m).resolved_config_json;
                    dataset_ids{j,m,r} = run.all_runs{r}(m).run_metadata.dataset_id;
                end
            end
            run.results = run.all_runs{1};
            values = double(run.mc_summary.final_param_estimates);
            states = run.mc_summary.statuses;
            estimates(j,:,:,:) = reshape(values,[1 M R n]);
            statuses(j,:,:) = reshape(states,[1 M R]);
            stats = study_bias_statistics(values,states,sys.theta_true,study.summary_population);
            for key = fieldnames(stats).'
                name = key{1}; value = stats.(name);
                if ismember(name,{'bias','variance','rmse','mcse','ci_low','ci_high','count'})
                    statistics.(name)(j,:,:) = reshape(value,[1 M n]);
                else
                    statistics.(name)(j,:) = reshape(value,[1 M]);
                end
            end
            if field_or(study,'save_case_results',true)
                [~,stem] = fileparts(output);
                case_file = fullfile(folder,sprintf('%s_sigma%02d.mat',stem,j));
                run.study = study; run.study.type = 'noise_case';
                run.study.sigma_nu = levels(j); run.study.noise_level_index = j;
                save(case_file,'-struct','run','-v7');
                case_files{j} = case_file;
            end
        end
        saved = study;
        saved.sigma_nu_values = levels; saved.innovation_variances = levels.^2;
        saved.method_names = methods; saved.record_length = length;
        saved.theta_true = sys.theta_true(:);
        saved.param_labels = run.experiment.param_labels; saved.param_names = run.experiment.param_names;
        saved.final_param_estimates = estimates; saved.statuses = statuses;
        saved.dataset_ids = dataset_ids; saved.resolved_configs = resolved_configs;
        saved.data_seeds = cfg.seed+(0:R-1); saved.case_files = case_files;
        saved.data_pairing = 'matched_input_and_standard_innovations';
        saved.initial_condition_policy = 'zero_state_fit_after_burn_in';
        saved.config_json = jsonencode(study);
        for key = fieldnames(statistics).', saved.(key{1}) = statistics.(key{1}); end
        payload = struct('schema_version',2,'parametrization','PL','study',saved, ...
            'experiment_name',experiment_name,'resolved_experiment_json',jsonencode(cfg));
    case 'record_length'
        lengths = study.record_lengths(:).';
        assert(~isempty(lengths) && all(isfinite(lengths)) && all(lengths>=1) && ...
            all(lengths==fix(lengths)) && all(diff(lengths)>0), ...
            'record_lengths must be positive, unique, increasing integers.');
        theta0 = parameter_vector(study.theta0,n,center);
        assert(is_stable(theta0,sys,1),'theta0 denominators must be stable.');
        cfg.initialization = struct('policy','consistent','theta0',theta0);
        cfg.mode = 'MONTECARLO'; cfg.seed = study.data_seed;
        progress = study_progress('PL record_length',cfg.selected_methods,R,lengths,1,study);
        if progress.enabled, base_options.progress_callback = progress.callback; end
        progress.stage(sprintf('Preparing %d independent records of N=%d',R,lengths(end)));
        full_records = cell(R,1);
        for r = 1:R
            full_records{r} = reference_data(cfg,lengths(end),cfg.seed+r-1);
            progress.stage(sprintf('Prepared data record %d/%d',r,R),r==R);
        end
        methods = cellstr(string(cfg.selected_methods)); methods = methods(:).';
        M = numel(methods); L = numel(lengths);
        estimates = NaN(1,L,M,R,n); statuses = cell(1,L,M,R);
        resolved_configs = cell(1,L,M,R); dataset_ids = cell(1,L,M,R);
        statistics = struct();
        for j = 1:L
            cfg.settings.lambda_g = lengths(j);
            data_runs = cell(R,1);
            for r = 1:R
                full = full_records{r};
                data_runs{r} = struct('r',full.r(1:lengths(j)), ...
                    'nu',full.nu(1:lengths(j)),'c',full.c(1:lengths(j)));
            end
            run_options = base_options;
            run_options.data_runs = data_runs;
            run_options.save_results = field_or(study,'save_case_results',false);
            [~,stem] = fileparts(output);
            run_options.output_file = fullfile(folder,sprintf('%s_n%d.mat',stem,lengths(j)));
            run = comparison(cfg,run_options);
            progress.stage(sprintf('Summarizing recursive_PL, N=%d',lengths(j)));
            values = double(run.mc_summary.final_param_estimates);
            states = run.mc_summary.statuses;
            estimates(1,j,:,:,:) = reshape(values,[1 1 M R n]);
            statuses(1,j,:,:) = reshape(states,[1 1 M R]);
            for r = 1:R
                for m = 1:M
                    resolved_configs{1,j,m,r} = run.all_runs{r}(m).resolved_config_json;
                    dataset_ids{1,j,m,r} = run.all_runs{r}(m).run_metadata.dataset_id;
                end
            end
            stats = study_bias_statistics(values,states,sys.theta_true,study.summary_population);
            for key = fieldnames(stats).'
                name = key{1}; value = stats.(name);
                if ismatrix(value) && size(value,2)==n
                    statistics.(name)(1,j,:,:) = reshape(value,[1 1 M n]);
                else
                    statistics.(name)(1,j,:) = reshape(value,[1 1 M]);
                end
            end
        end
        coefficients = NaN(1,M,n,2);
        for m = 1:M
            for p = 1:n
                y = reshape(statistics.bias(1,:,m,p),[],1); keep = isfinite(y);
                if nnz(keep)>=3
                    fit = [ones(nnz(keep),1) 1./lengths(keep).']\y(keep);
                    coefficients(1,m,p,:) = reshape(fit,[1 1 1 2]);
                end
            end
        end
        saved = study;
        saved.type = 'record_length'; saved.resolution_labels = {'recursive_PL'};
        saved.method_names = methods; saved.theta_true = sys.theta_true(:); saved.theta0 = theta0;
        saved.param_labels = run.experiment.param_labels; saved.param_names = run.experiment.param_names;
        saved.final_param_estimates = estimates; saved.statuses = statuses;
        saved.dataset_ids = dataset_ids; saved.resolved_configs = resolved_configs;
        saved.data_seeds = cfg.seed+(0:R-1); saved.inverse_n_coefficients = coefficients;
        saved.initial_condition_policy = 'zero_state_fit_after_burn_in';
        saved.config_json = jsonencode(study);
        for key = fieldnames(statistics).', saved.(key{1}) = statistics.(key{1}); end
        payload = struct('schema_version',2,'parametrization','PL','study',saved, ...
            'experiment_name',experiment_name,'resolved_experiment_json',jsonencode(cfg));
    otherwise
        error('Unknown study: %s',kind);
end
progress.stage('Saving study results');
save(output,'-struct','payload','-v7');
progress.finish(output);
end

function starts = draw_starts(study,sys,default_center)
n = numel(default_center); R = study.n_runs;
center = parameter_vector(study.center,n,default_center);
scale = parameter_vector(study.scale,n,[]);
assert(all(scale>=0) && any(scale>0),'scale must be nonnegative and nonzero.');
radius = study.stability_radius;
assert(isscalar(radius) && radius>0 && radius<1,'stability_radius must lie in (0,1).');
assert(ismember(study.distribution,{'uniform','normal'}),'Unknown initialization distribution.');
stream = RandStream('mt19937ar','Seed',study.initialization_seed);
starts = zeros(n,R); accepted = 0;
for draw = 1:study.max_draws
    if strcmp(study.distribution,'uniform'), z=2*rand(stream,n,1)-1; else, z=randn(stream,n,1); end
    candidate = center+scale.*z;
    if is_stable(candidate,sys,radius)
        accepted = accepted+1; starts(:,accepted)=candidate;
        if accepted==R, return; end
    end
end
error('Could not draw enough stable starts; change center/scale or increase max_draws.');
end

function values = parameter_vector(value,n,default)
if isempty(value), value=default; end
values = value(:); if isscalar(values), values=repmat(values,n,1); end
assert(numel(values)==n && all(isfinite(values)),'Expected %d finite parameters.',n);
end

function tf = is_stable(theta,sys,radius)
a = theta(1:sys.na); d = theta(sys.na+sys.nb+sys.nf:end);
tf = all(abs(roots([1;a]))<radius) && all(abs(roots([1;d]))<radius);
end

function data = reference_data(cfg,length,seed)
s = cfg.system; theta = s.theta_true(:); burn = cfg.settings.burn_in;
stream = RandStream('mt19937ar','Seed',seed);
u = randn(stream,length+burn,1); nu = cfg.settings.sigma_nu*randn(stream,length+burn,1);
a = theta(1:s.na); b = theta(s.na+1:s.na+s.nb);
alpha = filter([0;b].',[1;a].',u); beta=alpha;
f = theta(s.na+s.nb+1:s.na+s.nb+s.nf-1);
for k = 1:numel(f), beta=beta+f(k)*alpha.^(k+1); end
d = theta(s.na+s.nb+s.nf:end); c=beta+filter(1,[1;d].',nu);
keep = burn+(1:length); data=struct('r',u(keep),'nu',nu(keep),'c',c(keep));
end

function noise_progress(progress,event,sigma)
if ~strcmp(event.phase,'stage'), event.metadata.sigma_nu = sigma; end
progress.callback(event);
end

function value = field_or(s,key,default)
if isfield(s,key), value=s.(key); else, value=default; end
end
