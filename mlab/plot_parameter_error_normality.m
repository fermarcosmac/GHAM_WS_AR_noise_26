function [fig, report, info] = plot_parameter_error_normality(dataset, fig_dir, prefix, export_figures, method_names, colors, options)
% Histograms of Monte Carlo parameter errors, with Shapiro-Wilk test results.
if nargin<7, options=struct(); end
[report,info]=parameter_error_normality(dataset,options);
errors=double(dataset.mc_summary.final_errors);
if size(errors,1)~=height(report), errors=errors.'; end
states=repmat("unrecorded",size(errors));
if isfield(dataset.mc_summary,'statuses')
    states=string(dataset.mc_summary.statuses);
    if size(states,1)~=height(report), states=states.'; end
end
valid=isfinite(errors) & ismember(states,["ok","converged","stalled","max_iterations","unrecorded"]);
M=height(report); n_cols=3; n_rows=ceil(M/n_cols);
fig=figure('Name',sprintf('%s %s Monte Carlo error normality',string(dataset.experiment.name),dataset.parametrization), ...
    'Color','w','Units','centimeters','Position',[2 2 30 7.5*n_rows]);
tl=tiledlayout(fig,n_rows,n_cols,'TileSpacing','compact','Padding','compact');
legend_added=false;
for m=1:M
    ax=nexttile(tl); hold(ax,'on');
    x=errors(m,valid(m,:));
    color_index=find(strcmp(method_names,char(report.Optimizer(m))),1);
    color=[.2 .45 .7]; if ~isempty(color_index), color=colors(color_index,:); end
    if isempty(x)
        text(ax,.5,.5,'No valid observations / unavailable','Units','normalized', ...
            'HorizontalAlignment','center','Color','k','FontSize',8);
        axis(ax,'off');
    else
        bars=histogram(ax,x,'Normalization','pdf','NumBins',max(5,ceil(sqrt(numel(x)))), ...
            'FaceColor',color,'FaceAlpha',.6,'EdgeColor','w');
        mu=report.MeanError(m); sigma=report.StdError(m);
        if isfinite(sigma) && sigma>0
            pad=.08*max(max(x)-min(x),sigma);
            xx=linspace(min(x)-pad,max(x)+pad,300);
            density=exp(-.5*((xx-mu)/sigma).^2)/(sigma*sqrt(2*pi));
            curve=plot(ax,xx,density,'k-','LineWidth',1.2);
            if ~legend_added
                lg=legend(ax,[bars curve],{'Observed errors','Fitted normal density'}, ...
                    'Interpreter','none','TextColor','k','Color','w','Orientation','horizontal','FontSize',9);
                lg.Layout.Tile='south'; legend_added=true;
            end
        end
        grid(ax,'on'); box(ax,'on');
        if isfinite(report.W(m))
            note=sprintf('n = %d | W = %.4f\np = %.3g | p_H = %.3g\n%s', ...
                report.ValidSamples(m),report.W(m),report.RawP(m),report.AdjustedP(m),report.Decision(m));
        else
            note=sprintf('n = %d | %s',report.ValidSamples(m),report.Decision(m));
        end
        % Keep annotations above the tallest bar/curve, avoiding hidden data.
        limits=ylim(ax); ylim(ax,[0 limits(2)*1.38]);
        text(ax,.98,.98,note,'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top', ...
            'Interpreter','none','FontSize',8,'Color','k','BackgroundColor','w','Margin',1);
    end
    set(ax,'Color','w','XColor','k','YColor','k','GridColor',[.7 .7 .7],'FontSize',8);
    title(ax,report.Label(m),'Interpreter','none','Color','k','FontSize',10);
end
for m=M+1:n_rows*n_cols, axis(nexttile(tl),'off'); end
title(tl,{sprintf('%s | %s | Monte Carlo final parameter errors',string(dataset.experiment.name),dataset.parametrization), ...
    sprintf('Shapiro-Wilk; Holm alpha = %.3g; no outlier filtering; non-rejection does not establish normality',info.Alpha)}, ...
    'Interpreter','none','Color','k','FontSize',10);
xlabel(tl,'Final relative parameter error (%)','Interpreter','none','Color','k');
ylabel(tl,'Probability density','Interpreter','none','Color','k');
if export_figures
    if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
    stem=fullfile(fig_dir,['parameter_error_normality_' prefix]);
    exportgraphics(fig,[stem '.pdf'],'ContentType','vector');
    exportgraphics(fig,[stem '.png'],'Resolution',150);
    writetable(report,[stem '.csv']);
    fid=fopen([stem '_metadata.json'],'w'); assert(fid~=-1,'Cannot open normality metadata.');
    cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(info,'PrettyPrint',true));
    fprintf('Exported Shapiro-Wilk histograms: %s (%d/%d reject normality after Holm).\n', ...
        stem,sum(report.RejectHolm),info.FamilySize);
end
end
