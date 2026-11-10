function reports = export_montecarlo_paper_tables(output_dir)
% Regenerate manuscript tables from saved results; never rerun identification.
% Uses the plotting loader to share BI-HMC and recover complete wall times.
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<1, output_dir=fullfile(root,'output','paper_tables'); end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
reports=struct();
for example={'example_1','example_CSTR'}
    name=example{1};
    datasets=plot_comparison_results(struct('experiment_name',name, ...
        'mode','MONTECARLO','include_initialization',false,'statistics_only',true, ...
        'export_figures',false,'show_figures',false, ...
        'win_test',struct('enabled',false),'normality',struct('enabled',false)));
    assert(numel(datasets)==2,'Expected both PL and FSM saved comparisons.');
    rows=cell(0,14); fragments=cell(1,2);
    for p=1:2
        backend={'PL','FSM'};
        dataset=datasets(strcmpi({datasets.parametrization},backend{p}));
        assert(isscalar(dataset),'Expected one saved comparison per parametrization.');
        s=dataset.mc_summary; names=cellstr(string(s.method_names(:)));
        M=numel(names); errors=method_matrix(s.final_errors,M);
        output=method_matrix(s.final_output_rmse_runs,M);
        times=method_matrix(s.total_wall_time_runs,M);
        assert(isequal(size(errors),size(output),size(times)),'Incompatible saved run axes.');
        R=size(errors,2); values=NaN(M,6); counts=zeros(M,3);
        order=[find(~strcmp(names,'BI_HMC')).' find(strcmp(names,'BI_HMC')).'];
        for m=order
            [values(m,1),values(m,2),counts(m,1)]=finite_stats(errors(m,:));
            [values(m,3),values(m,4),counts(m,2)]=finite_stats(output(m,:));
            [values(m,5),values(m,6),counts(m,3)]=finite_stats(times(m,:));
            states=string(s.statuses(m,:)); kinds=unique(states,'stable');
            statuses=strjoin(arrayfun(@(v)v+":"+sum(states==v),kinds),'; ');
            rows(end+1,:)={backend{p},method_label(names{m},backend{p}),names{m}, ...
                char(statuses),counts(m,1),values(m,1),values(m,2), ...
                counts(m,2),values(m,3),values(m,4),counts(m,3), ...
                values(m,5),values(m,6),strcmp(backend{p},'PL') && strcmp(names{m},'BI_HMC')}; %#ok<AGROW>
        end
        if strcmp(name,'example_1'), title_name='Example~1'; tag='example-1';
        else, title_name='CSTR'; tag='cstr'; end
        active=any(counts(:,1:2)>0,2);
        assert(all(counts(active,1)==R & counts(active,2)==R & counts(active,3)==R), ...
            'Incomplete method results: add explicit counts before reporting all runs.');
        warning_note='';
        bi_index=find(strcmp(names,'BI_HMC'),1);
        if ~isempty(bi_index)
            warnings=sum(string(s.statuses(bi_index,:))=="sampling_warning");
            if warnings>0
                warning_note=sprintf([' BI-HMC sampler diagnostics flagged %d of %d fits; ' ...
                    'their finite point estimates are included.'],warnings,R);
            end
        end
        text={ ...
            '\begin{table}[tbp]', '\centering', '\small', ...
            '\setlength{\tabcolsep}{4pt}', ...
            sprintf(['\\caption{%s Monte Carlo results (%s parametrization). Entries are mean ' ...
                '$\\pm$ sample standard deviation over %d runs. Timing is total wall-clock time; ' ...
                'BI-HMC includes warm-up and sampling and is the same saved Python reference ' ...
                'in both parametrizations. Skipped methods are omitted.%s ' ...
                'Bold means indicate column minima at the displayed four-decimal precision.}'], ...
                title_name,backend{p},R,warning_note), ...
            sprintf('\\label{tab:%s-montecarlo-%s}',tag,lower(backend{p})), ...
            '\begin{tabular}{lrrr}', '\toprule', ...
            'Method & Parameter error (\%) & Output RMSE & Wall time (s) \\', '\midrule'};
        rounded_means=round(values(:,[1 3 5]),4);
        best_means=min(rounded_means(active,:),[],1);
        for m=order
            if ~active(m), continue; end
            cells=cell(1,3);
            for c=1:3
                mean_text=sprintf('%.4f',values(m,2*c-1));
                if rounded_means(m,c)==best_means(c)
                    mean_text=['\mathbf{' mean_text '}'];
                end
                cells{c}=sprintf('$%s \\pm %.4f$',mean_text,values(m,2*c));
            end
            text{end+1}=sprintf('%s & %s & %s & %s \\\\', ...
                method_label(names{m},backend{p}),cells{:}); %#ok<AGROW>
        end
        text=[text {'\bottomrule','\end{tabular}','\end{table}'}]; %#ok<AGROW>
        fragments{p}=strjoin(text,newline);
    end
    report=cell2table(rows,'VariableNames',{'parametrization','method','method_id','statuses', ...
        'parameter_error_pct_n','parameter_error_pct_mean','parameter_error_pct_std', ...
        'output_rmse_n','output_rmse_mean','output_rmse_std', ...
        'wall_time_s_n','wall_time_s_mean','wall_time_s_std','shared_bi_hmc_reference'});
    % The Bayesian row is a shared reference, not a separate PL computation.
    bi=report(strcmp(report.method_id,'BI_HMC'),:);
    assert(height(bi)==2 && isequaln(bi{1,5:13},bi{2,5:13}), ...
        'BI-HMC statistics must be identical in the PL and FSM tables.');
    file=fullfile(output_dir,[name '_montecarlo_tables.tex']);
    fid=fopen(file,'w'); assert(fid>=0,'Cannot write %s.',file);
    cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,'%% Requires \\usepackage{booktabs}.\n');
    fprintf(fid,'%% Generated by mlab/export_montecarlo_paper_tables.m.\n');
    fprintf(fid,'%% Sources: results/%s/comparison_{pl,fsm}_montecarlo.mat.\n',name);
    fprintf(fid,'%% All finite final estimates included; sample SD uses n-1.\n');
    fprintf(fid,'%% Complete wall times recovered by plot_comparison_results from all_runs.\n');
    fprintf(fid,'%s\n\n%s\n',fragments{:});
    clear cleanup;
    writetable(report,fullfile(output_dir,[name '_montecarlo_summary.csv']));
    reports.(name)=report;
    fprintf('Exported paper tables: %s\n',file);
end
end

function matrix=method_matrix(values,M)
matrix=double(values);
if size(matrix,1)~=M, matrix=matrix.'; end
assert(size(matrix,1)==M,'Expected one row per method.');
end

function [mu,sd,n]=finite_stats(values)
values=values(isfinite(values)); n=numel(values); mu=NaN; sd=NaN;
if n>0, mu=mean(values); end
if n>1, sd=std(values,0); end
end

function label=method_label(name,backend)
label=method_figure_label(name);
label=strrep(label,'RGLS','RLS');
if strcmp(backend,'FSM')
    if strcmp(label,'WS-GNI'), label='Newton'; end
    if strcmp(label,'WS-GGI'), label='GI'; end
end
end
