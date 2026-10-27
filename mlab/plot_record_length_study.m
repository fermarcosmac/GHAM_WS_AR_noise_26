function [figures, outcome_report, aggregate_report] = plot_record_length_study(payload, options)
% Plot parameter bias/RMSE, outcomes, and aggregate bias by optimizer.
% The optional second output is a numeric table, also exported as a separate CSV.
% Aggregate figures are appended after the existing three figures per resolution.
if nargin<2, options=struct(); end
if ischar(payload) || isstring(payload), payload=load(payload); end
s = payload.study; backend=char(payload.parametrization);
names=cellstr(string(s.method_names)); labels=cellstr(string(s.param_labels));
resolutions=cellstr(string(s.resolution_labels)); lengths=double(s.record_lengths(:).');
M=numel(names); P=numel(s.theta_true); L=numel(lengths); C=numel(resolutions);
bias=reshape(s.bias,[C L M P]); se=reshape(s.mcse,[C L M P]); rmse=reshape(s.rmse,[C L M P]);
finite=reshape(s.finite_count,[C L M]); converged=reshape(s.converged_count,[C L M]);
failed=reshape(s.failed_count,[C L M]); skipped=reshape(s.skipped_count,[C L M]);
stalled=NaN(C,L,M); limited=NaN(C,L,M);
if isfield(s,'statuses')
    states=reshape(string(s.statuses),[C L M double(s.n_runs)]);
    stalled=sum(states=="stalled",4); limited=sum(states=="max_iterations",4);
end
if isfield(s,'selected_count')
    used=reshape(s.selected_count,[C L M]);
elseif strcmp(char(s.summary_population),'converged')
    used=converged;
else
    used=finite;
end
outcome_report=build_outcome_report(backend,resolutions,lengths,names,double(s.n_runs), ...
    finite,converged,stalled,limited,failed,skipped,used,char(s.summary_population));
export=isfield(options,'export_figures') && options.export_figures;
folder=fullfile(fileparts(fileparts(mfilename('fullpath'))),'figs');
if isfield(options,'fig_dir'), folder=options.fig_dir; end
if export && ~exist(folder,'dir'), mkdir(folder); end
colors=lines(M); figures=gobjects(0);
for c=1:C
    for metric={'bias','rmse'}
        field=metric{1};
        fig=figure('Color','w','Name',[backend ' record-length ' field], ...
            'Units','centimeters','Position',[2 2 30 8.5*ceil(P/3)]);
        tl=tiledlayout(fig,ceil(P/3),3,'TileSpacing','compact','Padding','compact');
        for p=1:P
            ax=nexttile(tl); hold(ax,'on'); ax.TickLabelInterpreter='latex';
            for m=1:M
                y=reshape(bias(c,:,m,p),1,L);
                if strcmp(field,'bias')
                    errorbar(ax,lengths,y,1.96*reshape(se(c,:,m,p),1,L),'-o', ...
                        'Color',colors(m,:),'DisplayName',latex_literal(method_figure_label(names{m})));
                else
                    plot(ax,lengths,reshape(rmse(c,:,m,p),1,L),'-o', ...
                        'Color',colors(m,:),'DisplayName',latex_literal(method_figure_label(names{m})));
                end
            end
            ax.XScale='log'; grid(ax,'on'); box(ax,'on');
            if p>3*(ceil(P/3)-1)
                xlabel(ax,'Record length $N$','Interpreter','latex');
            end
            if strcmp(field,'bias')
                yline(ax,0,'k--','HandleVisibility','off');
                % Keep tight automatic ranges, but cap extreme-value axes.
                limits=ylim(ax);
                bounded_limits=[max(limits(1),-1),min(limits(2),1)];
                if bounded_limits(1)>=bounded_limits(2), bounded_limits=[-1 1]; end
                ylim(ax,bounded_limits);
                if mod(p-1,3)==0, ylabel(ax,'Signed mean error','Interpreter','latex'); end
            else
                ax.YScale='log';
                if mod(p-1,3)==0, ylabel(ax,'Parameter RMSE','Interpreter','latex'); end
            end
            title(ax,['$' labels{p} '$'],'Interpreter','latex');
            if p==1
                lg=legend(ax,'Interpreter','latex','NumColumns',3); lg.Layout.Tile='south';
            end
        end
        if strcmp(field,'bias')
            title_text=sprintf('%s / %s: mean error; population=%s; 95%% MC intervals', ...
                backend,resolutions{c},char(s.summary_population));
        else
            title_text=sprintf('%s / %s: parameter RMSE; population=%s', ...
                backend,resolutions{c},char(s.summary_population));
        end
        title(tl,latex_literal(title_text),'Interpreter','latex');
        figures(end+1)=fig; %#ok<AGROW>
        save_figure(fig,folder,payload,resolutions{c},field,export);
    end
    table_rows=outcome_report(strcmp(outcome_report.Resolution,resolutions{c}),:);
    fig=plot_outcome_table(table_rows,backend,resolutions{c},M,char(s.summary_population));
    figures(end+1)=fig; %#ok<AGROW>
    save_figure(fig,folder,payload,resolutions{c},'outcomes',export);
end
fprintf('\n%s record-length outcomes (counts out of Trials; percentages are in the table figures/CSV):\n',backend);
disp(outcome_report(:,{'Resolution','N','Method','Trials','Finite','Converged', ...
    'Stalled','MaxIterations','Failed','Skipped','Used'}));
if export
    writetable(outcome_report,fullfile(folder, ...
        ['record_length_outcomes_' char(payload.experiment_name) '_' lower(backend) '.csv']));
    rows=cell(C*L*M*P,15); i=0;
    counts=reshape(s.count,[C L M P]); variance=reshape(s.variance,[C L M P]);
    for c=1:C
        for j=1:L
            for m=1:M
                for p=1:P
                    i=i+1; b=bias(c,j,m,p); e=se(c,j,m,p);
                    rows(i,:)={backend,resolutions{c},lengths(j),names{m},labels{p}, ...
                        b,e,b-1.96*e,b+1.96*e,variance(c,j,m,p),rmse(c,j,m,p), ...
                        counts(c,j,m,p),finite(c,j,m),converged(c,j,m),failed(c,j,m)+skipped(c,j,m)};
                end
            end
        end
    end
    report=cell2table(rows,'VariableNames',{'Parametrization','Resolution','N','Method','Parameter', ...
        'Bias','MCSE','CILow','CIHigh','Variance','RMSE','Used','Finite','Converged','FailedOrSkipped'});
    writetable(report,fullfile(folder,['record_length_' char(payload.experiment_name) '_' lower(backend) '.csv']));
end
[aggregate_figures,aggregate_report]=plot_record_length_aggregate_bias(payload,options);
figures=[figures aggregate_figures.'];
end

function report=build_outcome_report(backend,resolutions,lengths,names,trials, ...
    finite,converged,stalled,limited,failed,skipped,used,population)
rows=cell(numel(resolutions)*numel(lengths)*numel(names),21); i=0;
for c=1:numel(resolutions)
    for j=1:numel(lengths)
        for m=1:numel(names)
            i=i+1;
            counts=double([finite(c,j,m),converged(c,j,m),stalled(c,j,m), ...
                limited(c,j,m),failed(c,j,m),skipped(c,j,m),used(c,j,m)]);
            values=reshape([counts;100*counts/trials],1,[]);
            rows(i,:)=[{backend,resolutions{c},lengths(j),method_figure_label(names{m}), ...
                names{m},trials},num2cell(values),{population}];
        end
    end
end
report=cell2table(rows,'VariableNames',{'Parametrization','Resolution','N','Method','MethodID', ...
    'Trials','Finite','FinitePct','Converged','ConvergedPct','Stalled','StalledPct', ...
    'MaxIterations','MaxIterationsPct','Failed','FailedPct','Skipped','SkippedPct', ...
    'Used','UsedPct','SummaryPopulation'});
end

function fig=plot_outcome_table(rows,backend,resolution,methods_per_length,population)
% Draw text and grid lines so the complete table exports as vector PDF.
n=height(rows); header_height=1.6;
fig=figure('Color','w','Name',[backend ' record-length outcomes table: ' resolution], ...
    'Units','centimeters','Position',[2 2 34 max(12,4+0.47*(n+header_height))]);
annotation(fig,'textbox',[.02 .945 .96 .04],'String', ...
    latex_literal([backend ' / ' resolution ': optimization outcomes']), ...
    'Interpreter','latex','FontWeight','bold','FontSize',11,'EdgeColor','none');
annotation(fig,'textbox',[.02 .905 .96 .035],'String', ...
    latex_literal(['Population: ' population '. Outcome cells show count (% of all attempted trials).']), ...
    'Interpreter','latex','FontSize',9,'EdgeColor','none');
ax=axes(fig,'Position',[.02 .095 .96 .795]); hold(ax,'on');
xlim(ax,[0 1]); ylim(ax,[0 n+header_height]); axis(ax,'off'); box(ax,'on');
widths=[.06 .15 .045 repmat(.105,1,7)]; edges=[0 cumsum(widths/sum(widths))];
headers={'N','Method','Trials','Finite','Converged','Stalled', ...
    'Max iterations','Failed / invalid','Skipped','Used'};
fields={'Finite','Converged','Stalled','MaxIterations','Failed','Skipped','Used'};
rectangle(ax,'Position',[0 n 1 header_height],'FaceColor',[.89 .92 .96],'EdgeColor','none');
for i=1:n
    y=n-i;
    if mod(floor((i-1)/methods_per_length),2)==0
        rectangle(ax,'Position',[0 y 1 1],'FaceColor',[.96 .97 .98],'EdgeColor','none');
    end
    for k=1:10
        align='center'; color=[.1 .1 .1]; weight='normal';
        if k==1, value=sprintf('%d',rows.N(i));
        elseif k==2, value=char(string(rows.Method(i))); align='left';
        elseif k==3, value=sprintf('%d',rows.Trials(i));
        else
            field=fields{k-3}; count=rows.(field)(i);
            if isnan(count)
                value='n/a';
            else
                value=sprintf('%d (%.1f%%)',count,rows.([field 'Pct'])(i));
            end
            if count==0, color=[.5 .5 .5]; end
            if strcmp(field,'Failed') && count>0
                rectangle(ax,'Position',[edges(k) y edges(k+1)-edges(k) 1], ...
                    'FaceColor',[1 .90 .90],'EdgeColor','none');
                color=[.65 .08 .08]; weight='bold';
            end
        end
        x=(edges(k)+edges(k+1))/2;
        if k==2, x=edges(k)+.006; end
        text(ax,x,y+.5,latex_literal(value),'HorizontalAlignment',align,'VerticalAlignment','middle', ...
            'Interpreter','latex','FontSize',8.5,'Color',color,'FontWeight',weight);
    end
end
for k=1:10
    text(ax,(edges(k)+edges(k+1))/2,n+header_height/2,headers{k}, ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'Interpreter','latex','FontSize',8.5,'FontWeight','bold');
end
for x=edges
    plot(ax,[x x],[0 n+header_height],'-','Color',[.65 .65 .65],'LineWidth',.4);
end
for y=[0:n n+header_height]
    width=.35;
    if y==n || mod(n-y,methods_per_length)==0, width=.8; end
    plot(ax,[0 1],[y y],'-','Color',[.65 .65 .65],'LineWidth',width);
end
annotation(fig,'textbox',[.02 .005 .96 .075],'String',{ ...
    'Finite includes converged, stalled, and iteration-limited estimates. Used is the population included in bias/RMSE summaries.', ...
    'Failed includes invalid updates (red); skipped methods are separate. n/a means the saved file lacks that status breakdown.'}, ...
    'Interpreter','latex','FontSize',8.5,'EdgeColor','none');
end

function save_figure(fig,folder,payload,resolution,metric,export)
if ~export, return; end
tag=regexprep([char(payload.experiment_name) '_' lower(char(payload.parametrization)) '_' resolution],'[^a-zA-Z0-9_-]','_');
exportgraphics(fig,fullfile(folder,['record_length_' metric '_' tag '.pdf']),'ContentType','vector');
end

function value=latex_literal(value)
% Escape literal identifiers and percentages without changing math labels.
value=strrep(char(value),'_','\_');
value=strrep(value,'%','\%');
end
