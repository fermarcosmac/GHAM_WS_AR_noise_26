function figures = plot_bayesian_posteriors(dataset, options)
% Plot retained marginal posteriors for the representative saved record.
% Separate datasets/initializations are never pooled into one posterior.
% posterior_samples accepts [chain,draw,parameter] or [draw,parameter].
if nargin<2, options=struct(); end
figures=gobjects(0);
truth=double(dataset.experiment.theta_true(:)); P=numel(truth);
labels=cellstr(string(dataset.experiment.param_labels));
results=dataset.results;
if iscell(results), results=[results{:}]; end
for m=1:numel(results)
    result=results(m);
    if ~is_bayesian_result(result) || ~isfield(result,'posterior_samples') || ...
            isempty(result.posterior_samples)
        continue;
    end
    samples=posterior_matrix(result.posterior_samples,P);
    samples=samples(all(isfinite(samples),2),:);
    if isempty(samples), continue; end
    variance=[];
    if isfield(result,'posterior_variance_samples')
        variance=double(result.posterior_variance_samples(:));
        variance=variance(isfinite(variance) & variance>0);
    end
    total=P+~isempty(variance); ncols=min(3,total); nrows=ceil(total/ncols);
    fig=figure('Name',sprintf('%s %s posterior distributions', ...
        dataset.parametrization,char(string(result.name))), ...
        'Color','w','Units','centimeters','Position',[2 2 25 6.6*nrows]);
    if isprop(fig,'Theme'), fig.Theme='light'; end
    tl=tiledlayout(fig,nrows,ncols,'TileSpacing','compact','Padding','compact');
    point=double(result.theta_hat(:));
    if numel(point)~=P, point=mean(samples,1).'; end
    for p=1:total
        ax=nexttile(tl); hold(ax,'on');
        if p<=P
            values=samples(:,p); reference=truth(p);
            estimate=point(p); label=labels{p};
        else
            values=variance; reference=NaN;
            if isfield(dataset.experiment,'sigma_nu')
                reference=double(dataset.experiment.sigma_nu)^2;
            end
            estimate=mean(values); label='\sigma_\nu^2';
        end
        histogram(ax,values,'Normalization','pdf','FaceColor',[.12 .42 .68], ...
            'FaceAlpha',.32,'EdgeColor','none','DisplayName','Retained posterior', ...
            'Tag','bayesian_posterior');
        interval=prctile(values,[2.5 97.5]);
        xline(ax,interval(1),':','Color',[.28 .28 .28], ...
            'DisplayName','95\% credible interval','Tag','posterior_interval');
        xline(ax,interval(2),':','Color',[.28 .28 .28],'HandleVisibility','off', ...
            'Tag','posterior_interval');
        if isfinite(reference)
            xline(ax,reference,'k--','DisplayName','True value','Tag','true_parameter');
        end
        if isfinite(estimate)
            xline(ax,estimate,'-','Color',[.75 .22 .12], ...
                'DisplayName','Posterior point estimate','Tag','posterior_point');
        end
        title(ax,['$' label '$'],'Interpreter','latex');
        xlabel(ax,['$' label '$'],'Interpreter','latex');
        ylabel(ax,'Posterior density','Interpreter','latex');
        grid(ax,'on'); box(ax,'on'); ax.TickLabelInterpreter='latex';
        if p==1
            lg=legend(ax,'Interpreter','latex','FontSize',8,'NumColumns',2);
            lg.Layout.Tile='south';
        end
    end
    title(tl,sprintf('%s / %s: posterior distributions',dataset.parametrization, ...
        method_figure_label(char(string(result.name)))),'Interpreter','none');
    subtitle(tl,{sprintf('%d retained draws; representative record; warm-up excluded', ...
        size(samples,1)),['Sampler status: ' char(string(result.status))]}, ...
        'Interpreter','none','FontSize',9);
    figures(end+1)=fig; %#ok<AGROW>
    if isfield(options,'export_figures') && options.export_figures
        folder=char(string(options.fig_dir));
        if ~exist(folder,'dir'), mkdir(folder); end
        stem=regexprep(char(string(result.name)),'[^a-zA-Z0-9_-]','_');
        prefix=char(string(options.prefix));
        exportgraphics(fig,fullfile(folder,['posterior_' prefix '_' stem '.pdf']), ...
            'ContentType','vector');
    end
end
end

function samples=posterior_matrix(raw,P)
samples=double(raw);
if P==1
    samples=reshape(samples,[],1);
elseif ismatrix(samples) && size(samples,2)==P
    return;
elseif size(samples,ndims(samples))==P
    samples=reshape(samples,[],P);
elseif ismatrix(samples) && size(samples,1)==P
    samples=samples.';
else
    error('plot_bayesian_posteriors:Shape', ...
        'Expected [chain,draw,parameter] or [draw,parameter] posterior samples.');
end
end
