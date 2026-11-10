function test_study_single_method()
% Regression: summary axes survive one method and multiple parameter columns.
root=fileparts(fileparts(mfilename('fullpath')));
validation_root=char(java.io.File(fullfile(root,'.validation')).getCanonicalPath());
folder=tempname(validation_root); mkdir(folder);
cleanup=onCleanup(@()cleanup_test(folder,validation_root)); %#ok<NASGU>
cfg=jsondecode(fileread(fullfile(root,'configs','example_1.json')));
cfg.settings.lambda_g=32; cfg.settings.burn_in=4; cfg.settings.K_max=2;
cfg.settings.conv_threshold=0; cfg.diagnostics.enabled=false;
cfg.selected_methods={'WS-GGI'};
n=numel(cfg.system.theta_true); R=2; L=3;
options=struct('experiment_cfg',cfg,'study',struct('n_runs',R,'progress',false, ...
    'record_lengths',[16 24 32],'save_case_results',false), ...
    'output_file',fullfile(folder,'record.mat'));
b=record_length_study('example_1',options);
assert(isequal(size(b.study.bias),[1 L 1 n]));
for j=1:L
    values=reshape(b.study.final_param_estimates(1,j,:,:,:),[1 R n]);
    states=reshape(b.study.statuses(1,j,:,:),[1 R]);
    expected=study_bias_statistics(values,states,cfg.system.theta_true,'finite');
    for key={'bias','variance','rmse','mcse','ci_low','ci_high','count'}
        field=key{1}; actual=reshape(b.study.(field)(1,j,1,:),[1 n]);
        assert(isequaln(actual,expected.(field)),'Record summary field %s differs.',field);
    end
end
for p=1:n
    y=reshape(b.study.bias(1,:,1,p),[],1);
    expected=[ones(L,1) 1./b.study.record_lengths(:)]\y;
    actual=reshape(b.study.inverse_n_coefficients(1,1,p,:),[],1);
    assert(max(abs(actual-expected))<1e-12);
end
options.study=struct('n_runs',R,'progress',false, ...
    'sigma_nu_values',[.05 .1 .2],'save_case_results',false);
options.output_file=fullfile(folder,'noise.mat');
b=noise_study('example_1',options);
assert(isequal(size(b.study.bias),[L 1 n]));
for j=1:L
    values=reshape(b.study.final_param_estimates(j,:,:,:),[1 R n]);
    states=reshape(b.study.statuses(j,:,:),[1 R]);
    expected=study_bias_statistics(values,states,cfg.system.theta_true,'finite');
    assert(isequaln(reshape(b.study.bias(j,1,:),[1 n]),expected.bias));
end
fprintf('MATLAB single-method record-length/noise statistics checks passed.\n');
end

function cleanup_test(folder,validation_root)
resolved=char(java.io.File(folder).getCanonicalPath());
assert(strcmp(fileparts(resolved),validation_root),'Unexpected cleanup path.');
if exist(resolved,'dir'), rmdir(resolved,'s'); end
end
