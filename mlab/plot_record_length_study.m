function figures = plot_record_length_study(payload, options)
% Plot signed bias with pointwise normal MC intervals, RMSE, and outcome rates.
if nargin<2, options=struct(); end
if ischar(payload) || isstring(payload), payload=load(payload); end
s = payload.study; backend=char(payload.parametrization);
names=cellstr(string(s.method_names)); labels=cellstr(string(s.param_labels));
resolutions=cellstr(string(s.resolution_labels)); lengths=double(s.record_lengths(:).');
M=numel(names); P=numel(s.theta_true); L=numel(lengths); C=numel(resolutions);
bias=reshape(s.bias,[C L M P]); se=reshape(s.mcse,[C L M P]); rmse=reshape(s.rmse,[C L M P]);
finite=reshape(s.finite_count,[C L M]); converged=reshape(s.converged_count,[C L M]);
failed=reshape(s.failed_count,[C L M])+reshape(s.skipped_count,[C L M]);
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
            ax=nexttile(tl); hold(ax,'on');
            for m=1:M
                y=reshape(bias(c,:,m,p),1,L);
                if strcmp(field,'bias')
                    errorbar(ax,lengths,y,1.96*reshape(se(c,:,m,p),1,L),'-o', ...
                        'Color',colors(m,:),'DisplayName',method_figure_label(names{m}));
                else
                    plot(ax,lengths,reshape(rmse(c,:,m,p),1,L),'-o', ...
                        'Color',colors(m,:),'DisplayName',method_figure_label(names{m}));
                end
            end
            ax.XScale='log'; grid(ax,'on'); xlabel(ax,'Record length N');
            if strcmp(field,'bias'), yline(ax,0,'k--','HandleVisibility','off'); ylabel(ax,'Signed mean error');
            else, ylabel(ax,'Parameter RMSE'); end
            title(ax,labels{p},'Interpreter','none');
            if p==1
                lg=legend(ax,'Interpreter','none','NumColumns',3); lg.Layout.Tile='south';
            end
        end
        title(tl,sprintf('%s / %s: %s, population=%s; intervals = MC uncertainty of mean', ...
            backend,resolutions{c},field,char(s.summary_population)),'Interpreter','none');
        figures(end+1)=fig; %#ok<AGROW>
        save_figure(fig,folder,payload,resolutions{c},field,export);
    end
    fig=figure('Color','w','Name',[backend ' record-length outcomes'], ...
        'Units','centimeters','Position',[2 2 30 12]);
    tl=tiledlayout(fig,1,3,'TileSpacing','compact');
    rates={finite,converged,failed}; titles={'Finite outcomes','Converged','Failed or skipped'};
    for k=1:3
        ax=nexttile(tl); hold(ax,'on'); values=rates{k};
        for m=1:M
            plot(ax,lengths,reshape(values(c,:,m),1,L)/double(s.n_runs),'-o', ...
                'Color',colors(m,:),'DisplayName',method_figure_label(names{m}));
        end
        ax.XScale='log'; ylim(ax,[0 1]); grid(ax,'on'); title(ax,titles{k}); xlabel(ax,'Record length N');
        if k==1
            lg=legend(ax,'Interpreter','none','NumColumns',3); lg.Layout.Tile='south';
        end
    end
    title(tl,[backend ' / ' resolutions{c} ': outcome fractions'],'Interpreter','none');
    figures(end+1)=fig; %#ok<AGROW>
    save_figure(fig,folder,payload,resolutions{c},'outcomes',export);
end
if export
    rows=cell(C*L*M*P,15); i=0;
    counts=reshape(s.count,[C L M P]); variance=reshape(s.variance,[C L M P]);
    for c=1:C
        for j=1:L
            for m=1:M
                for p=1:P
                    i=i+1; b=bias(c,j,m,p); e=se(c,j,m,p);
                    rows(i,:)={backend,resolutions{c},lengths(j),names{m},labels{p}, ...
                        b,e,b-1.96*e,b+1.96*e,variance(c,j,m,p),rmse(c,j,m,p), ...
                        counts(c,j,m,p),finite(c,j,m),converged(c,j,m),failed(c,j,m)};
                end
            end
        end
    end
    report=cell2table(rows,'VariableNames',{'Parametrization','Resolution','N','Method','Parameter', ...
        'Bias','MCSE','CILow','CIHigh','Variance','RMSE','Used','Finite','Converged','FailedOrSkipped'});
    writetable(report,fullfile(folder,['record_length_' char(payload.experiment_name) '_' lower(backend) '.csv']));
end
end

function save_figure(fig,folder,payload,resolution,metric,export)
if ~export, return; end
tag=regexprep([char(payload.experiment_name) '_' lower(char(payload.parametrization)) '_' resolution],'[^a-zA-Z0-9_-]','_');
exportgraphics(fig,fullfile(folder,['record_length_' metric '_' tag '.pdf']),'ContentType','vector');
end
