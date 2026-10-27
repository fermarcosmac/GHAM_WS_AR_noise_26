function [figures, report] = plot_record_length_aggregate_bias(payload, options)
% One optimizer per panel: Euclidean norm of the signed mean-error vector.
% This is norm(mean(error)), not mean(norm(error)) or total parameter RMSE.
% Uses the saved summary population without rerunning any identification.
if nargin<2, options=struct(); end
if ischar(payload) || isstring(payload), payload=load(payload); end
s=payload.study;
names=cellstr(string(s.method_names));
resolutions=cellstr(string(s.resolution_labels));
lengths=double(s.record_lengths(:).');
C=numel(resolutions); L=numel(lengths); M=numel(names); P=numel(s.theta_true);
bias=reshape(s.bias,[C L M P]);
counts=reshape(s.count,[C L M P]);
assert(all(counts==counts(:,:,:,1),'all'), ...
    'All parameters must use the same retained realizations.');
used=counts(:,:,:,1);
% No omission of missing parameters: an unavailable bias vector stays NaN.
total_bias=NaN(C,L,M);
for c=1:C
    for j=1:L
        for m=1:M
            total_bias(c,j,m)=norm(reshape(bias(c,j,m,:),P,1),2);
        end
    end
end
export=isfield(options,'export_figures') && options.export_figures;
folder=fullfile(fileparts(fileparts(mfilename('fullpath'))),'figs');
if isfield(options,'fig_dir'), folder=options.fig_dir; end
if export && ~exist(folder,'dir'), mkdir(folder); end
colors=lines(M); figures=gobjects(C,1);
rows=cell(C*L*M,8); row=0;
for c=1:C
    fig=figure('Color','w','Name',[char(payload.parametrization) ...
        ' record-length aggregate bias / ' resolutions{c}], ...
        'Units','centimeters','Position',[2 2 30 7.5*ceil(M/3)]);
    tl=tiledlayout(fig,ceil(M/3),3,'TileSpacing','compact','Padding','compact');
    for m=1:M
        ax=nexttile(tl); hold(ax,'on'); ax.TickLabelInterpreter='latex';
        y=reshape(total_bias(c,:,m),1,L);
        plot(ax,lengths,y,'-o','Color',colors(m,:),'LineWidth',1.1, ...
            'MarkerSize',4,'Tag','aggregate_bias');
        ax.XScale='log'; grid(ax,'on'); box(ax,'on');
        xlim(ax,[lengths(1) lengths(end)]);
        title(ax,method_figure_label(names{m}),'Interpreter','none');
        if m>3*(ceil(M/3)-1)
            xlabel(ax,'Record length $N$','Interpreter','latex');
        end
        if mod(m-1,3)==0
            ylabel(ax,'$\|\widehat{\boldsymbol b}_N\|_2$','Interpreter','latex');
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
                    'MarkerSize',5,'Clipping','off','Tag','clipped_bias');
            end
            retained=reshape(used(c,:,m),1,L);
            if any(retained<double(s.n_runs))
                count_text=strjoin(arrayfun(@(n)sprintf('%d',n),retained, ...
                    'UniformOutput',false),', ');
                text(ax,.02,.95,['Retained: ' count_text ' / ' num2str(s.n_runs)], ...
                    'Units','normalized','VerticalAlignment','top', ...
                    'Interpreter','none','FontSize',7.5,'BackgroundColor','white');
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
                names{m},total_bias(c,j,m),used(c,j,m),double(s.n_runs),char(s.summary_population)};
        end
    end
    title(tl,[char(payload.parametrization) ' / ' strrep(resolutions{c},'_','\_') ...
        ': norm of mean parameter error'],'Interpreter','latex');
    subtitle(tl,{'All parameters; original parameter scale; no Monte Carlo intervals', ...
        'Linear y axes capped at 1; triangles mark clipped points'}, ...
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
    'Method','BiasNorm','Used','Trials','SummaryPopulation'});
if export
    writetable(report,fullfile(folder,['record_length_aggregate_bias_' ...
        char(payload.experiment_name) '_' lower(char(payload.parametrization)) '.csv']));
end
end
