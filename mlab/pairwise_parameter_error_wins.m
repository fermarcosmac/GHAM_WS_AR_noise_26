function [pairs, trials, info] = pairwise_parameter_error_wins(dataset, options)
% Exact paired sign tests of final relative parameter error, with Holm control.
% One family = one study, benchmark, and parametrization. No outlier filtering.
% Win probabilities and their exact 95% CIs condition on valid, non-tied pairs;
% the CIs are descriptive (unadjusted), while decisions use Holm-adjusted p.
if nargin < 2, options = struct(); end
alpha = opt(options,'alpha',0.05);
tie_ulps = opt(options,'tie_ulps',10);
absolute_tie_tolerance = opt(options,'absolute_tie_tolerance',0);
expected_runs = opt(options,'expected_runs',30);
assert(isscalar(alpha) && alpha>0 && alpha<1,'alpha must lie in (0,1).');
assert(isscalar(tie_ulps) && isfinite(tie_ulps) && tie_ulps>=0,'Invalid tie_ulps.');
assert(isscalar(absolute_tie_tolerance) && isfinite(absolute_tie_tolerance) && ...
    absolute_tie_tolerance>=0,'Invalid absolute_tie_tolerance.');
summary = dataset.mc_summary;
names = string(summary.method_names(:));
assert(numel(unique(names))==numel(names),'Optimizer names must be unique.');
errors = orient(double(summary.final_errors),numel(names));
assert(~any(errors(isfinite(errors))<0),'Parameter errors must be nonnegative.');
[M,R] = size(errors);
if ~isempty(expected_runs) && R~=expected_runs
    warning('pairwise_parameter_error_wins:RunCount','Expected %d runs; found %d. Using the saved count.',expected_runs,R);
end
states = repmat("unrecorded",M,R);
if isfield(summary,'statuses')
    states = orient(string(summary.statuses),M);
    assert(isequal(size(states),size(errors)),'Statuses and errors must align.');
end
valid = isfinite(errors) & ismember(states,["ok","converged","stalled","max_iterations","unrecorded"]);
skipped = all(states=="skipped",2);
% Legacy Monte Carlo payloads record statuses only for their first run.
if ~isfield(summary,'statuses') && isfield(dataset,'results')
    results = dataset.results;
    if iscell(results), results = [results{:}]; end
    for m = 1:M
        idx = find(string({results.name})==names(m),1);
        if ~isempty(idx) && strcmpi(string(results(idx).status),'skipped') && ~any(valid(m,:))
            skipped(m) = true;
        end
    end
end
study = lower(string(dataset.experiment.mode));
if isfield(dataset,'study') && isfield(dataset.study,'type'), study = lower(string(dataset.study.type)); end
assert(ismember(study,["montecarlo","initialization"]),'Unsupported study for win comparisons.');
benchmark = string(dataset.experiment.name);
parametrization = string(dataset.parametrization);
pairing = check_pairing(dataset,names,R,study);
labels = names;
if isfield(options,'method_labels')
    labels = string(options.method_labels(:));
    assert(numel(labels)==M,'method_labels must match the saved method order.');
end
active = find(~skipped);
rows = struct([]); trial_rows = struct([]);
for ia = 1:numel(active)
    a = active(ia);
    for ib = ia+1:numel(active)
        b = active(ib);
        joint = valid(a,:) & valid(b,:);
        x = errors(a,joint); y = errors(b,joint);
        tolerance = max(absolute_tie_tolerance,tie_ulps*eps(max(1,max(abs(x),abs(y)))));
        d = x-y;
        wins = sum(d < -tolerance); losses = sum(d > tolerance);
        ties = sum(abs(d)<=tolerance); decisive = wins+losses;
        p = NaN; probability = NaN; ci = [NaN NaN];
        conclusion = "no valid pairs";
        if any(joint)
            p = 1; conclusion = "all ties";
            if decisive>0
                p = min(1,2*binocdf(min(wins,losses),decisive,0.5));
                [probability,ci] = binofit(wins,decisive,alpha);
                conclusion = "inconclusive";
            end
        end
        entry = struct('Benchmark',benchmark,'Study',study,'Parametrization',parametrization, ...
            'OptimizerA',names(a),'LabelA',labels(a),'OptimizerB',names(b),'LabelB',labels(b), ...
            'TotalRuns',R,'ValidPairs',sum(joint),'WinsA',wins,'WinsB',losses,'Ties',ties, ...
            'Decisive',decisive,'AInvalidOnly',sum(~valid(a,:) & valid(b,:)), ...
            'BInvalidOnly',sum(valid(a,:) & ~valid(b,:)),'BothInvalid',sum(~valid(a,:) & ~valid(b,:)), ...
            'WinProbabilityA',probability,'CILower',ci(1),'CIUpper',ci(2), ...
            'MeanDifference',mean(d),'MedianDifference',median(d), ...
            'RawP',p,'AdjustedP',NaN,'Significant',false,'Winner',"", ...
            'Conclusion',conclusion,'PairingCheck',pairing);
        if isempty(rows), rows = entry; else, rows(end+1) = entry; end %#ok<AGROW>
        outcomes = repmat("both_invalid",1,R);
        outcomes(~valid(a,:) & valid(b,:)) = "A_invalid";
        outcomes(valid(a,:) & ~valid(b,:)) = "B_invalid";
        joint_outcomes = repmat("tie",1,numel(d));
        joint_outcomes(d < -tolerance) = "A_win";
        joint_outcomes(d > tolerance) = "B_win";
        outcomes(joint) = joint_outcomes;
        for r = 1:R
            t = struct('Benchmark',benchmark,'Study',study,'Parametrization',parametrization, ...
                'OptimizerA',names(a),'OptimizerB',names(b),'Run',r,'ErrorA',errors(a,r), ...
                'ErrorB',errors(b,r),'StatusA',states(a,r),'StatusB',states(b,r),'Outcome',outcomes(r));
            if isempty(trial_rows), trial_rows = t; else, trial_rows(end+1) = t; end %#ok<AGROW>
        end
    end
end
pairs = struct2table(rows); trials = struct2table(trial_rows);
if ~isempty(pairs)
    raw = pairs.RawP;
    raw(~isfinite(raw)) = 1; % Retain untestable applicable pairs in family size.
    [sorted,order] = sort(raw);
    adjusted = min(1,cummax((numel(raw):-1:1)'.*sorted));
    pairs.AdjustedP(order) = adjusted;
    pairs.AdjustedP(~isfinite(pairs.RawP)) = NaN;
    pairs.Significant = pairs.AdjustedP<alpha;
    for k = find(pairs.Significant).'
        if pairs.WinsA(k)>pairs.WinsB(k), pairs.Winner(k)=pairs.OptimizerA(k);
        else, pairs.Winner(k)=pairs.OptimizerB(k); end
        pairs.Conclusion(k) = "significant win-frequency advantage";
    end
end
info = struct('Benchmark',benchmark,'Study',study,'Parametrization',parametrization, ...
    'Metric',"100*norm(theta_hat-theta_true,2)/norm(theta_true,2)", ...
    'Alpha',alpha,'TieULPs',tie_ulps,'AbsoluteTieTolerance',absolute_tie_tolerance, ...
    'Runs',R,'MethodNames',names,'MethodLabels',labels,'FiniteCounts',sum(valid,2), ...
    'Skipped',skipped,'FamilySize',height(pairs),'PairingCheck',pairing, ...
    'ConfidenceIntervals',"unadjusted exact binomial; conditional on valid non-tied pairs");
if isfield(dataset,'file'), info.SourceFile = string(dataset.file); end
if isfield(dataset,'generated_at'), info.SourceGeneratedAt = string(dataset.generated_at); end
end

function matrix = orient(matrix,M)
if size(matrix,1)==M, return; end
if size(matrix,2)==M, matrix=matrix.'; return; end
error('pairwise_parameter_error_wins:Shape','Expected one row per optimizer.');
end

function status = check_pairing(dataset,names,R,study)
status = "run_index_only";
if ~isfield(dataset,'all_runs') || isempty(dataset.all_runs)
    warning('pairwise_parameter_error_wins:LegacyPairing', ...
        'No per-run metadata for %s/%s/%s: pairing uses saved run columns.', ...
        string(dataset.experiment.name),string(dataset.parametrization),study);
    return;
end
runs = dataset.all_runs;
assert(iscell(runs) && numel(runs)==R,'Run histories and summary columns must align.');
ids = strings(R,1); complete = true;
for r = 1:R
    results = runs{r};
    if iscell(results), results=[results{:}]; end
    reference_id = ""; reference_start = []; reference_hash = "";
    for m = 1:numel(names)
        k = find(string({results.name})==names(m),1);
        if isempty(k), complete=false; continue; end
        result = results(k);
        if ~isfield(result,'run_metadata') || ~isfield(result.run_metadata,'dataset_id')
            complete=false; continue;
        end
        md = result.run_metadata;
        id = string(md.dataset_id);
        assert(strlength(id)>0,'Empty dataset identifier.');
        if reference_id=="", reference_id=id;
        else, assert(id==reference_id,'Optimizers do not share a data realization within run %d.',r); end
        if isfield(md,'mc_index'), assert(double(md.mc_index)==r,'Run index mismatch.'); end
        if isfield(md,'initialization_index'), assert(double(md.initialization_index)==r,'Initialization index mismatch.'); end
        if isfield(dataset.mc_summary,'run_ids') && isfield(md,'run_id')
            assert(string(dataset.mc_summary.run_ids(m,r))==string(md.run_id),'Summary/history run ID mismatch.');
        end
        if study=="initialization"
            if isfield(result,'theta0')
                start=double(result.theta0(:));
                if isempty(reference_start), reference_start=start;
                else, assert(isequal(start,reference_start),'Optimizers do not share the starting point.'); end
                if isfield(dataset.study,'initializations')
                    assert(isequal(start,double(dataset.study.initializations(:,r))),'Saved starting point mismatch.');
                end
            else, complete=false;
            end
            if isfield(md,'initialization_sha256')
                hash=string(md.initialization_sha256);
                if reference_hash=="", reference_hash=hash;
                else, assert(hash==reference_hash,'Initialization hash mismatch.'); end
            end
        end
    end
    ids(r)=reference_id;
end
if complete
    if study=="montecarlo"
        assert(numel(unique(ids))==R,'Monte Carlo data realizations are repeated.');
    else
        assert(numel(unique(ids))==1,'Initialization study does not use a fixed data record.');
    end
    status="metadata_validated";
else
    status="partially_validated";
    warning('pairwise_parameter_error_wins:PartialPairing','Incomplete per-run metadata; some pairing checks unavailable.');
end
end

function value = opt(options,key,default)
if isfield(options,key), value=options.(key); else, value=default; end
end
