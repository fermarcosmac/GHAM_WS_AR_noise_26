function [fig, pairs, info] = plot_parameter_error_wins(dataset, fig_dir, prefix, export_figures, options)
% Plot/export paired wins and independent comparisons for shared references.
if nargin<5, options=struct(); end
[pairs,trials,info] = pairwise_parameter_error_wins(dataset,options);
active = find(~info.Skipped); M=numel(active);
fig = figure('Name',sprintf('%s %s %s parameter-error wins',info.Benchmark,info.Study,info.Parametrization), ...
    'Color','w','Units','centimeters','Position',[2 2 32 25]);
ax=axes(fig,'Color',[.92 .92 .92],'XColor','k','YColor','k');
if M<2
    text(ax,.5,.5,'Fewer than two applicable methods','Units','normalized','Color','k');
    axis(ax,'off');
else
    names=info.MethodNames(active); labels=info.MethodLabels(active);
    values=NaN(M); captions=strings(M); bold=false(M);
    for k=1:height(pairs)
        a=find(names==pairs.OptimizerA(k)); b=find(names==pairs.OptimizerB(k));
        if ~isfinite(pairs.RawP(k))
            captions(a,b)="N/A"; captions(b,a)="N/A"; continue;
        end
        values(a,b)=0; values(b,a)=0;
        if pairs.Significant(k)
            direction=2*double(pairs.Winner(k)==pairs.OptimizerA(k))-1;
            values(a,b)=direction; values(b,a)=-direction;
            bold(a,b)=true; bold(b,a)=true;
        end
        captions(a,b)=sprintf('%d:%d:%d\np=%.2g',pairs.WinsA(k),pairs.WinsB(k),pairs.Ties(k),pairs.AdjustedP(k));
        captions(b,a)=sprintf('%d:%d:%d\np=%.2g',pairs.WinsB(k),pairs.WinsA(k),pairs.Ties(k),pairs.AdjustedP(k));
        if pairs.Test(k)=="independent_rank_sum"
            captions(a,b)=sprintf('P=%.2f\np=%.2g (RS)',pairs.WinProbabilityA(k),pairs.AdjustedP(k));
            captions(b,a)=sprintf('P=%.2f\np=%.2g (RS)',1-pairs.WinProbabilityA(k),pairs.AdjustedP(k));
        end
    end
    im=imagesc(ax,values,[-1 1]); im.AlphaData=isfinite(values);
    colormap(ax,[.96 .78 .73; .97 .97 .97; .70 .85 .96]);
    xticks(ax,1:M); yticks(ax,1:M);
    xticklabels(ax,labels); yticklabels(ax,compose('%s [%d/%d]',labels,info.FiniteCounts(active),info.Runs));
    xtickangle(ax,40);
    set(ax,'TickLabelInterpreter','none','FontSize',8,'Color',[.92 .92 .92],'XColor','k','YColor','k');
    axis(ax,'image'); box(ax,'on');
    for a=1:M
        for b=1:M
            caption=captions(a,b); if a==b, caption="--"; end
            weight='normal'; if bold(a,b), weight='bold'; end
            text(ax,b,a,caption,'HorizontalAlignment','center','Color','k','Interpreter','none', ...
                'FontSize',7,'FontWeight',weight);
        end
    end
    cb=colorbar(ax,'Ticks',[-1 0 1],'TickLabels',{'Column superior','Inconclusive','Row superior'},'Color','k');
    cb.TickLabelInterpreter='none';
end
test_note=sprintf('Paired exact sign tests; Holm alpha = %.3g; cells = row wins:losses:ties and adjusted p',info.Alpha);
population_note='Accuracy conditional on jointly valid outputs; ties excluded from test; labels show valid outputs/attempts';
if ~isempty(pairs) && any(pairs.Test=="independent_rank_sum")
    test_note=sprintf('Paired sign / independent rank-sum (RS); Holm alpha = %.3g',info.Alpha);
    population_note='RS: P = cross-sample row-win probability (half credit for ties); all valid samples; labels = valid/attempted';
end
title(ax,{sprintf('%s | %s | %s | %d trials',info.Benchmark,info.Study,info.Parametrization,info.Runs), ...
    test_note}, ...
    'Interpreter','none','Color','k','FontSize',10);
xlabel(ax,population_note, ...
    'Interpreter','none','Color','k','FontSize',8);
if export_figures
    if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
    stem=fullfile(fig_dir,['parameter_error_wins_' prefix]);
    exportgraphics(fig,[stem '.pdf'],'ContentType','vector');
    writetable(pairs,[stem '.csv']);
    writetable(trials,[stem '_trials.csv']);
    fid=fopen([stem '_metadata.json'],'w'); assert(fid~=-1,'Cannot open metadata file.');
    cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(info,'PrettyPrint',true));
    significant_count=0; if ~isempty(pairs), significant_count=sum(pairs.Significant); end
    fprintf('Exported parameter-error tests: %s (%d/%d significant comparisons).\n',stem,significant_count,height(pairs));
end
end
