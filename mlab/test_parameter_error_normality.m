function test_parameter_error_normality()
% Fixed reference sample plus normal quantiles and undefined-test cases.
values=NaN(5,30);
values(1,1:11)=[148 154 158 160 161 162 166 170 182 195 236];
values(2,:)=5+norminv(((1:30)-.5)/30);
values(3,:)=1;
values(5,1:2)=[1 2];
ds=struct('experiment',struct('name','fixture','mode','MONTECARLO'), ...
    'parametrization','FSM','mc_summary',struct('method_names',{{'skew','quantiles','constant','empty','short'}}, ...
    'final_errors',values));
[r,info]=parameter_error_normality(ds);
assert(info.FamilySize==2 && info.Engine=="scipy.stats.shapiro");
assert(isequal(r.ValidSamples,[11;30;30;0;2]));
% Published Shapiro-Wilk example (W=0.7888, p=0.0067038), not Shapiro-Francia.
assert(abs(r.W(1)-.7888)<1e-4 && abs(r.RawP(1)-.0067038)<1e-7);
assert(abs(r.AdjustedP(1)-2*r.RawP(1))<1e-12 && r.RejectHolm(1));
assert(~r.RejectHolm(2) && r.Decision(2)=="Normality not rejected");
assert(all(isnan(r.W(3:5))) && all(isnan(r.RawP(3:5))));
assert(r.Decision(3)=="constant sample; test undefined");
assert(r.Decision(4)=="No valid observations" && r.Decision(5)=="insufficient samples");
% A finite number attached to a failed outcome must not enter the sample.
ds.mc_summary.statuses=repmat({'converged'},5,30);
ds.mc_summary.statuses(1,11)={'failed'};
r=parameter_error_normality(ds);
assert(r.ValidSamples(1)==10 && r.ExcludedSamples(1)==20);
% Shapiro-Wilk is invariant to positive rescaling and shifting.
ds.mc_summary=rmfield(ds.mc_summary,'statuses');
ds.mc_summary.final_errors=values*2+7;
r2=parameter_error_normality(ds);
assert(abs(r2.W(1)-.7888)<1e-4 && abs(r2.RawP(1)-.0067038)<1e-7);
fprintf('Shapiro-Wilk normality checks passed.\n');
end
