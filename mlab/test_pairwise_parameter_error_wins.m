function test_pairwise_parameter_error_wins()
% Statistical checks with known binomial probabilities and pairing failures.
ds = struct('experiment',struct('name','fixture','mode','MONTECARLO'), ...
    'parametrization','FSM','study',struct(),'all_runs',{{}},'results',struct([]));
ds.mc_summary = struct('method_names',{{'A','B','C'}}, ...
    'final_errors',[ones(1,30);2*ones(1,30);ones(1,30)], ...
    'statuses',{repmat({'converged'},3,30)});
for r=1:30
    results=struct([]);
    for m=1:3
        result=struct('name',ds.mc_summary.method_names{m},'theta0',[0;1], ...
            'run_metadata',struct('dataset_id',sprintf('record_%d',r),'mc_index',r));
        results=[results result]; %#ok<AGROW>
    end
    ds.all_runs{r}=results;
end
[p,t,info]=pairwise_parameter_error_wins(ds);
assert(info.PairingCheck=="metadata_validated" && info.FamilySize==3);
assert(height(t)==90 && all(p.TotalRuns==30));
ab=p(p.OptimizerA=="A" & p.OptimizerB=="B",:);
assert(ab.WinsA==30 && ab.WinsB==0 && ab.Ties==0);
assert(abs(ab.RawP-2^-29)<1e-20 && abs(ab.AdjustedP-3*2^-29)<1e-20);
assert(ab.Winner=="A" && ab.Significant && ab.CILower>.5);
ac=p(p.OptimizerA=="A" & p.OptimizerB=="C",:);
assert(ac.Ties==30 && ac.RawP==1 && ~ac.Significant && isnan(ac.WinProbabilityA));
% Independent reference implementation supplied by the statistics toolbox.
reference=signtest(ds.mc_summary.final_errors(1,:),ds.mc_summary.final_errors(2,:),'method','exact');
assert(abs(reference-ab.RawP)<1e-20);
changed=ds;
changed.mc_summary.final_errors(1,1)=NaN;
changed.mc_summary.final_errors(2,2)=NaN;
changed.mc_summary.final_errors(1:2,3)=NaN;
p=pairwise_parameter_error_wins(changed);
ab=p(p.OptimizerA=="A" & p.OptimizerB=="B",:);
assert(ab.ValidPairs==27 && ab.Decisive==27 && ab.WinsA==27);
assert(ab.AInvalidOnly==1 && ab.BInvalidOnly==1 && ab.BothInvalid==1);
assert(ab.ValidPairs+ab.AInvalidOnly+ab.BInvalidOnly+ab.BothInvalid==30);
% Extreme finite outcomes are retained, independently of boxplot filtering.
changed=ds; changed.mc_summary.final_errors(1,1)=1e6;
p=pairwise_parameter_error_wins(changed);
ab=p(p.OptimizerA=="A" & p.OptimizerB=="B",:);
assert(ab.WinsA==29 && ab.WinsB==1 && ab.ValidPairs==30);
% A tiny difference is a numerical tie by default, and an exact win at 0 ULPs.
changed=ds; changed.mc_summary.final_errors(2,:)=1+eps(1);
p=pairwise_parameter_error_wins(changed);
assert(p.Ties(1)==30);
p=pairwise_parameter_error_wins(changed,struct('tie_ulps',0));
assert(p.WinsA(1)==30);
% Swapping methods reverses wins, probabilities, and winner interpretation.
changed=ds; changed.mc_summary.final_errors=changed.mc_summary.final_errors([2 1 3],:);
p=pairwise_parameter_error_wins(changed);
ab=p(p.OptimizerA=="A" & p.OptimizerB=="B",:);
assert(ab.WinsA==0 && ab.WinsB==30 && ab.Winner=="B" && ab.WinProbabilityA==0);
changed=ds; changed.mc_summary.statuses(3,:)={'skipped'};
changed.mc_summary.final_errors(3,:)=NaN;
[p,~,info]=pairwise_parameter_error_wins(changed);
assert(height(p)==1 && info.Skipped(3) && p.AdjustedP==p.RawP);
% A supported method with no valid output stays in the comparison family.
changed.mc_summary.statuses(3,:)={'failed'};
[p,~,info]=pairwise_parameter_error_wins(changed);
assert(height(p)==3 && ~info.Skipped(3));
assert(all(isnan(p.RawP(p.ValidPairs==0))));
assert(p.AdjustedP(1)==3*p.RawP(1));
changed=ds;
changed.all_runs{1}(2).run_metadata.dataset_id='wrong_data';
must_fail(@()pairwise_parameter_error_wins(changed));
changed=ds;
changed.all_runs{2}=changed.all_runs{1};
must_fail(@()pairwise_parameter_error_wins(changed));
% Initialization pairing checks both fixed data and shared starts.
changed=ds; changed.experiment.mode='INITIALIZATION';
changed.study=struct('type','initialization','initializations',repmat([0;1],1,30));
for r=1:30
    for m=1:3, changed.all_runs{r}(m).run_metadata.dataset_id='fixed_record'; end
end
[~,~,info]=pairwise_parameter_error_wins(changed);
assert(info.PairingCheck=="metadata_validated");
changed.all_runs{1}(2).theta0=[1;1];
must_fail(@()pairwise_parameter_error_wins(changed));
fprintf('Paired parameter-error win tests passed.\n');
end

function must_fail(operation)
failed=false;
try, operation(); catch, failed=true; end
assert(failed,'Invalid pairing was accepted.');
end
