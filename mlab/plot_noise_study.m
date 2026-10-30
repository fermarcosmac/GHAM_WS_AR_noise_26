function [figures, report] = plot_noise_study(payload, options)
% Plot the saved noise sweep for PL or FSM; never rerun identification.
% One run: final errors/estimates. Repeated runs: means and 95% MC intervals.
if nargin<2, options=struct(); end
if ischar(payload) || isstring(payload), payload=load(payload); end
s=payload.study;
assert(strcmp(char(s.type),'noise'),'Expected a saved noise study.');
backend=char(payload.parametrization);
levels=double(s.sigma_nu_values(:).');
names=cellstr(string(s.method_names)); labels=cellstr(string(s.param_labels));
L=numel(levels); M=numel(names); R=double(s.n_runs); P=numel(s.theta_true);
truth=double(s.theta_true(:));
assert(norm(truth)>0,'Relative parameter error requires a nonzero true parameter vector.');
values=reshape(double(s.final_param_estimates),[L M R P]);
states=reshape(string(s.statuses),[L M R]);
population=char(s.summary_population);
assert(ismember(population,{'finite','converged'}),'Unknown summary population.');
runtime=NaN(L,M,R);
for field={'total_wall_time','total_wall_time_runs','total_time_runs'}
    if isfield(s,field{1}) && ~isempty(s.(field{1}))
        runtime=reshape(double(s.(field{1})),[L M R]); break;
    end
end
means=NaN(L,M,P); param_se=means;
error_mean=NaN(L,M); error_se=error_mean; used=zeros(L,M);
rows=cell(L*M,21); row=0;
for j=1:L
    for m=1:M
        trials=reshape(values(j,m,:,:),[R P]);
        status=reshape(states(j,m,:),[R 1]);
        keep=all(isfinite(trials),2) & ismember(status,["converged","stalled","max_iterations", ...
            "ok","sampling_complete","sampling_warning"]);
        if strcmp(population,'converged'), keep=keep & ismember(status,["converged","sampling_complete"]); end
        used(j,m)=sum(keep);
        if any(keep)
            errors=100*vecnorm(trials(keep,:)-truth.',2,2)/norm(truth);
            error_mean(j,m)=mean(errors);
            means(j,m,:)=reshape(mean(trials(keep,:),1),[1 1 P]);
            if used(j,m)>1
                error_se(j,m)=std(errors)/sqrt(used(j,m));
                param_se(j,m,:)=reshape(std(trials(keep,:),0,1)/sqrt(used(j,m)),[1 1 P]);
            end
        end
        failed=sum(~ismember(status,["converged","stalled","max_iterations","ok", ...
            "sampling_complete","sampling_warning","skipped"]));
        times=reshape(runtime(j,m,:),[R 1]); times=times(isfinite(times) & times>=0);
        mean_time=NaN; std_time=NaN;
        if ~isempty(times), mean_time=mean(times); end
        if numel(times)>1, std_time=std(times); end
        row=row+1;
        rows(row,:)={backend,levels(j),levels(j)^2,names{m},R,used(j,m), ...
            error_mean(j,m),error_se(j,m),error_mean(j,m)-1.96*error_se(j,m), ...
            error_mean(j,m)+1.96*error_se(j,m),sum(status=="converged"), ...
            sum(status=="stalled"),sum(status=="max_iterations"),failed,sum(status=="skipped"),population, ...
            sum(status=="sampling_complete"),sum(status=="sampling_warning"), ...
            mean_time,std_time,numel(times)};
    end
end
report=cell2table(rows,'VariableNames',{'Parametrization','SigmaNu','InnovationVariance', ...
    'Method','Trials','Used','MeanRelativeErrorPct','MCSE','CILow','CIHigh', ...
    'Converged','Stalled','MaxIterations','Failed','Skipped','SummaryPopulation', ...
    'SamplerDiagnosticsPassed','SamplingWarning','MeanTotalWallTimeSec', ...
    'StdTotalWallTimeSec','TimingAttempts'});
figures=gobjects(1,2);
if R==1
    error_label='Relative parameter error (%)';
    estimate_label='Final estimate';
    note='One run per method and noise level; no Monte Carlo intervals.';
else
    error_label='Mean relative parameter error (%)';
    estimate_label='Mean final estimate';
    note='Means with pointwise approximate 95% MC intervals where at least two runs are retained.';
end

fig=figure('Color','w','Name',[backend ' noise sensitivity: final parameter error'], ...
    'Units','centimeters','Position',[2 2 28 7*ceil(M/3)]);
if isprop(fig,'Theme'), fig.Theme='light'; end
colors=colororder(fig); colors=repmat(colors,ceil(M/size(colors,1)),1);
tl=tiledlayout(fig,ceil(M/3),3,'TileSpacing','compact','Padding','compact');
for m=1:M
    ax=nexttile(tl); hold(ax,'on');
    plot_curve(ax,levels,error_mean(:,m),error_se(:,m),colors(m,:),names{m},'noise_error');
    format_axis(ax,levels);
    title(ax,method_figure_label(names{m}),'Interpreter','none');
    xlabel(ax,'Innovation standard deviation $\sigma_\nu$','Interpreter','latex');
    ylabel(ax,error_label,'Interpreter','none');
    if ~any(isfinite(error_mean(:,m)))
        if all(states(:,m,:)=="skipped",'all'), message='Skipped';
        else, message='No retained estimates'; end
        text(ax,.5,.5,message,'Units','normalized','HorizontalAlignment','center','Interpreter','none');
    elseif any(used(:,m)<R)
        counts=strjoin(arrayfun(@(n)sprintf('%d',n),used(:,m).','UniformOutput',false),', ');
        subtitle(ax,sprintf('Retained by noise level: %s / %d',counts,R),'Interpreter','none','FontSize',8);
    end
end
title(tl,[backend ': sensitivity to innovation standard deviation'],'Interpreter','none');
subtitle(tl,{note,['Summary population: ' population]},'Interpreter','none','FontSize',9);
figures(1)=fig;

fig=figure('Color','w','Name',[backend ' noise sensitivity: final parameter estimates'], ...
    'Units','centimeters','Position',[2 2 24 7*ceil(P/3)]);
if isprop(fig,'Theme'), fig.Theme='light'; end
tl=tiledlayout(fig,ceil(P/3),3,'TileSpacing','compact','Padding','compact');
for p=1:P
    ax=nexttile(tl); hold(ax,'on');
    for m=1:M
        plot_curve(ax,levels,means(:,m,p),param_se(:,m,p),colors(m,:), ...
            method_figure_label(names{m}),'noise_estimate');
    end
    yline(ax,truth(p),'k--','HandleVisibility','off','Tag','true_parameter');
    format_axis(ax,levels);
    title(ax,['$' labels{p} '$'],'Interpreter','latex');
    xlabel(ax,'Innovation standard deviation $\sigma_\nu$','Interpreter','latex');
    ylabel(ax,estimate_label,'Interpreter','none');
    if p==1
        lg=legend(ax,'Interpreter','none','NumColumns',3,'FontSize',8);
        lg.Layout.Tile='south';
    end
end
title(tl,[backend ': final parameter estimates across noise levels'],'Interpreter','none');
subtitle(tl,{note,'Dashed lines show the true parameters.'},'Interpreter','none','FontSize',9);
figures(2)=fig;

if isfield(options,'export_figures') && options.export_figures
    folder=fullfile(fileparts(fileparts(mfilename('fullpath'))),'figs');
    if isfield(options,'fig_dir'), folder=options.fig_dir; end
    if ~exist(folder,'dir'), mkdir(folder); end
    tag=regexprep([char(payload.experiment_name) '_' lower(backend)],'[^a-zA-Z0-9_-]','_');
    exportgraphics(figures(1),fullfile(folder,['noise_error_' tag '.pdf']),'ContentType','vector');
    exportgraphics(figures(2),fullfile(folder,['noise_estimates_' tag '.pdf']),'ContentType','vector');
    writetable(report,fullfile(folder,['noise_summary_' tag '.csv']));
end
end

function plot_curve(ax,x,y,se,color,label,tag)
% Plot points separately: undefined single-run MCSE must not hide the curve.
y=y(:).'; half=1.96*se(:).';
plot(ax,x,y,'-o','Color',color,'LineWidth',1.1,'MarkerSize',4, ...
    'DisplayName',label,'Tag',tag);
keep=isfinite(y) & isfinite(half);
if any(keep)
    errorbar(ax,x(keep),y(keep),half(keep),'LineStyle','none','Color',color, ...
        'CapSize',4,'HandleVisibility','off','Tag',[tag '_interval']);
end
end

function format_axis(ax,levels)
grid(ax,'on'); box(ax,'on');
ax.XTick=levels;
ax.XTickLabel=arrayfun(@(v)sprintf('%g',v),levels,'UniformOutput',false);
ax.XTickLabelRotation=60;
padding=max(.05*(levels(end)-levels(1)),.02*max(1,levels(end)));
xlim(ax,[max(0,levels(1)-padding) levels(end)+padding]);
end
