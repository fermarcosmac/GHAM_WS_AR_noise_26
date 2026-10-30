function [figures, report] = plot_record_length_aggregate_bias(payload, options)
% One optimizer per panel: mean per-run Euclidean parameter error and 95% MC CI.
% Historical function/export names are retained for compatibility. The metric
% is mean(norm(error)): an estimation-error magnitude, not statistical bias.
% Uses the saved summary population without rerunning any identification.
if nargin<2, options=struct(); end
if ischar(payload) || isstring(payload), payload=load(payload); end
s=payload.study;
names=cellstr(string(s.method_names));
resolutions=cellstr(string(s.resolution_labels));
lengths=double(s.record_lengths(:).');
C=numel(resolutions); L=numel(lengths); M=numel(names); P=numel(s.theta_true);
R=double(s.n_runs);
estimates=reshape(s.final_param_estimates,[C L M R P]);
states=reshape(string(s.statuses),[C L M R]);
assert(ismember(char(s.summary_population),{'finite','converged'}), ...
    'summary_population must be finite or converged.');
mean_error=NaN(C,L,M); std_error=NaN(C,L,M); mcse=NaN(C,L,M);
ci_low=NaN(C,L,M); ci_high=NaN(C,L,M); used=zeros(C,L,M);
for c=1:C
    for j=1:L
        for m=1:M
            values=reshape(estimates(c,j,m,:,:),[R P]);
            keep=all(isfinite(values),2);
            if strcmp(char(s.summary_population),'converged')
                keep=keep & ismember(reshape(states(c,j,m,:),[R 1]), ...
                    ["converged","sampling_complete"]);
            end
            used(c,j,m)=sum(keep);
            if ~any(keep), continue; end
            % Aggregate within each run BEFORE averaging over runs.
            errors=values(keep,:)-reshape(s.theta_true,1,P);
            run_error=vecnorm(errors,2,2);
            mean_error(c,j,m)=mean(run_error);
            if used(c,j,m)>1
                std_error(c,j,m)=std(run_error,0);
                mcse(c,j,m)=std_error(c,j,m)/sqrt(used(c,j,m));
                ci_low(c,j,m)=mean_error(c,j,m)-1.96*mcse(c,j,m);
                ci_high(c,j,m)=mean_error(c,j,m)+1.96*mcse(c,j,m);
            end
        end
    end
end
export=isfield(options,'export_figures') && options.export_figures;
folder=fullfile(fileparts(fileparts(mfilename('fullpath'))),'figs');
if isfield(options,'fig_dir'), folder=options.fig_dir; end
if export && ~exist(folder,'dir'), mkdir(folder); end
colors=lines(M); figures=gobjects(C,1);
rows=cell(C*L*M,12); row=0;
for c=1:C
    fig=figure('Color','w','Name',[char(payload.parametrization) ...
        ' record-length mean overall error / ' resolutions{c}], ...
        'Units','centimeters','Position',[2 2 30 7.5*ceil(M/3)]);
    tl=tiledlayout(fig,ceil(M/3),3,'TileSpacing','compact','Padding','compact');
    for m=1:M
        ax=nexttile(tl); hold(ax,'on'); ax.TickLabelInterpreter='latex';
        y=reshape(mean_error(c,:,m),1,L);
        half_width=1.96*reshape(mcse(c,:,m),1,L);
        errorbar(ax,lengths,y,half_width,'-o','Color',colors(m,:),'LineWidth',1.1, ...
            'MarkerSize',4,'CapSize',4,'Tag','aggregate_error');
        ax.XScale='log'; grid(ax,'on'); box(ax,'on');
        xlim(ax,[lengths(1)/1.05 lengths(end)*1.05]);
        title(ax,method_figure_label(names{m}),'Interpreter','none');
        if m>3*(ceil(M/3)-1)
            xlabel(ax,'Record length $N$','Interpreter','latex');
        end
        if mod(m-1,3)==0
            ylabel(ax,'Mean overall error','Interpreter','latex');
        end
        if any(isfinite(y))
            % Match the earlier display cap, with a nonnegative norm axis.
            limits=ylim(ax);
            ylim(ax,[0 min(1,max(limits(2),eps))]);
            clipped=isfinite(y) & y>ax.YLim(2);
            if any(clipped)
                % Boundary triangles distinguish clipped points from missing fits.
                plot(ax,lengths(clipped),repmat(ax.YLim(2),1,sum(clipped)), ...
                    '^','Color',colors(m,:),'MarkerFaceColor',colors(m,:), ...
                    'MarkerSize',5,'Clipping','off','Tag','clipped_error');
            end
            retained=reshape(used(c,:,m),1,L);
            if any(retained<double(s.n_runs))
                count_text=strjoin(arrayfun(@(n)sprintf('%d',n),retained, ...
                    'UniformOutput',false),', ');
                subtitle(ax,['Retained: ' count_text ' / ' num2str(s.n_runs)], ...
                    'Interpreter','none','FontSize',7.5);
            end
        else
            ylim(ax,[0 1]);
            skipped=reshape(s.skipped_count,[C L M]);
            if all(skipped(c,:,m)==double(s.n_runs)), message='Skipped';
            else, message='No retained estimates'; end
            text(ax,.5,.5,message,'Units','normalized', ...
                'HorizontalAlignment','center','Interpreter','none','Color',[.4 .4 .4]);
        end
        for j=1:L
            row=row+1;
            rows(row,:)={char(payload.parametrization),resolutions{c},lengths(j), ...
                names{m},mean_error(c,j,m),std_error(c,j,m),mcse(c,j,m), ...
                ci_low(c,j,m),ci_high(c,j,m),used(c,j,m),R,char(s.summary_population)};
        end
    end
    title(tl,[char(payload.parametrization) ' / ' strrep(resolutions{c},'_','\_') ...
        ': mean per-run parameter-error norm'],'Interpreter','latex');
    subtitle(tl,{sprintf('%d parameters; original scale; pointwise approximate 95%% MC intervals',P), ...
        'Linear y axes within [0,1]; outlying means/intervals clipped; triangles mark clipped means'}, ...
        'Interpreter','none','FontSize',9);
    figures(c)=fig;
    if export
        tag=regexprep([char(payload.experiment_name) '_' ...
            lower(char(payload.parametrization)) '_' resolutions{c}], ...
            '[^a-zA-Z0-9_-]','_');
        exportgraphics(fig,fullfile(folder,['record_length_aggregate_bias_' tag '.pdf']), ...
            'ContentType','vector');
    end
end
report=cell2table(rows,'VariableNames',{'Parametrization','Resolution','N', ...
    'Method','MeanErrorNorm','StdErrorNorm','MCSE','CILow','CIHigh', ...
    'Used','Trials','SummaryPopulation'});
if export
    writetable(report,fullfile(folder,['record_length_aggregate_bias_' ...
        char(payload.experiment_name) '_' lower(char(payload.parametrization)) '.csv']));
end
end
