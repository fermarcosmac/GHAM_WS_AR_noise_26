function [figures, report] = plot_optimization_diagnostics(dataset, fig_dir, prefix, export_figures, method_names, method_colors)
% Schema 2 diagnostics. Old result files without histories are accepted.
% Curves use observed iterations only. Bands are finite-value median/IQR;
% singularities remain Inf in files and are retained in the CSV summary.
if nargin < 4, export_figures = false; end
if nargin < 3, prefix = 'comparison'; end
if nargin < 2, fig_dir = ''; end
figures = gobjects(0); report = table();
runs = {dataset.results};
if isfield(dataset,'all_runs') && ~isempty(dataset.all_runs), runs = dataset.all_runs; end
if ~iscell(runs), runs = {runs}; end
records = struct([]);
names = {};
for j = 1:numel(runs)
    results = runs{j};
    if iscell(results), results = [results{:}]; end
    runs{j} = results;
    for m = 1:numel(results)
        r = results(m);
        if ~isfield(r,'diagnostics'), continue; end
        names = union(names,{char(r.name)},'stable');
        s = summarize_optimization_diagnostics(r.diagnostics);
        entry = struct('method',string(r.name),'run',j,'status',string(r.status), ...
            'run_id',string(r.run_metadata.run_id),'accepted_updates',r.accepted_updates, ...
            'optimization_seconds',r.total_time,'diagnostic_seconds',r.diagnostic_time);
        fields = {'raw_condition_initial','raw_condition_final','raw_condition_peak', ...
            'solver_condition_initial','solver_condition_final','solver_condition_peak', ...
            'raw_sample_count','raw_singular_fraction','raw_negative_curvature_fraction', ...
            'solver_singular_fraction','solver_fallback_count','projected_count','rejected_count'};
        for f = 1:numel(fields)
            entry.(fields{f}) = NaN;
            if isfield(s,fields{f}), entry.(fields{f}) = s.(fields{f}); end
        end
        if isempty(records), records = entry; else, records(end+1) = entry; end %#ok<AGROW>
    end
end
if isempty(records), return; end
report = struct2table(records);
fprintf('\n%s optimization diagnostics (one row per run; Inf denotes numerical singularity):\n',dataset.parametrization);
disp(report(:,{'method','run','status','raw_condition_initial','raw_condition_final', ...
    'raw_condition_peak','raw_singular_fraction','raw_negative_curvature_fraction'}));
if export_figures
    if ~exist(fig_dir,'dir'), mkdir(fig_dir); end
    writetable(report,fullfile(fig_dir,['optimization_summary_' prefix '.csv']));
end
% One figure per parametrization, with the same method color on every axis.
if nargin < 5 || isempty(method_names), method_names = names; end
if nargin < 6 || isempty(method_colors)
    method_colors = diagnostic_colors(numel(method_names));
end
fig = figure('Name',[dataset.parametrization ' optimization diagnostics'], ...
    'Color','w','Position',[80 80 1550 850]);
figures = fig;
tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
title(tl,sprintf('%s: method comparison; finite medians and interquartile bands', ...
    dataset.parametrization),'Interpreter','none','Color','k');
axes_list = gobjects(1,6);
for k = 1:6
    axes_list(k) = nexttile(tl);
    hold(axes_list(k),'on');
end
set(axes_list,'Color','w','XColor','k','YColor','k', ...
    'GridColor',[.6 .6 .6],'MinorGridColor',[.75 .75 .75]);
method_handles = gobjects(0); labels = {};
condition_styles = false(1,2);
for m = 1:numel(names)
    selected = {};
    for j = 1:numel(runs)
        idx = find(strcmp({runs{j}.name},names{m}),1);
        if ~isempty(idx) && ~strcmp(runs{j}(idx).status,'skipped') && isfield(runs{j}(idx),'diagnostics')
            selected{end+1} = runs{j}(idx); %#ok<AGROW>
        end
    end
    if isempty(selected), continue; end
    color_idx = find(strcmp(method_names,names{m}),1);
    assert(~isempty(color_idx),'Missing diagnostic color for method %s.',names{m});
    color = method_colors(color_idx,:);
    method_handles(end+1) = plot(axes_list(1),NaN,NaN,'Color',color,'LineWidth',1.6); %#ok<AGROW>
    labels{end+1} = names{m}; %#ok<AGROW>
    draw_band(axes_list(1),history_matrix(selected,'est_loss_hist'),color,names{m},'-');
    condition_styles(1) = draw_band(axes_list(2),diagnostic_matrix(selected,'raw_condition'),color,names{m},'-') || condition_styles(1);
    draw_band(axes_list(3),diagnostic_matrix(selected,'solver_condition'),color,names{m},'-');
    condition_styles(2) = draw_band(axes_list(2),diagnostic_matrix(selected,'scaled_condition'),color,names{m},':') || condition_styles(2);
    draw_band(axes_list(4),diagnostic_matrix(selected,'raw_eigen_min'),color,names{m},'-');
    draw_band(axes_list(5),diagnostic_matrix(selected,'gradient_norm'),color,names{m},'-');
    draw_band(axes_list(6),diagnostic_matrix(selected,'applied_step_norm'),color,names{m},'-');

end
if isempty(method_handles)
    close(fig); figures = gobjects(0); return;
end
set(axes_list([1 2 3 5 6]),'YScale','log');
title(axes_list(1),'Objective','Interpreter','none');
decorate(axes_list(1),'State iteration','Objective');
title(axes_list(2),'Raw Hessian condition (finite samples)','Interpreter','none');
decorate(axes_list(2),'Evaluation iteration','2-norm condition');
title(axes_list(3),'Solver matrix condition (finite samples)','Interpreter','none');
decorate(axes_list(3),'Evaluation iteration','2-norm condition');
yline(axes_list(4),0,':k','HandleVisibility','off');
title(axes_list(4),'Raw signed curvature','Interpreter','none');
decorate(axes_list(4),'Evaluation iteration','Minimum eigenvalue');
title(axes_list(5),'Gradient norm','Interpreter','none');
decorate(axes_list(5),'Evaluation iteration','Norm');
title(axes_list(6),'Applied step norm','Interpreter','none');
decorate(axes_list(6),'Evaluation iteration','Norm');
lgd = legend(axes_list(1),method_handles,labels,'Interpreter','none', ...
    'Orientation','horizontal','NumColumns',min(4,numel(labels)),'FontSize',10, ...
    'TextColor','k','Color','w');
lgd.Layout.Tile = 'south';
for ax = axes_list
    ax.Title.Color = 'k';
end
if condition_styles(2)
    style_legend(axes_list(2),{'Raw Hessian','Scaled Hessian'}, ...
        {'-',':'},condition_styles);
end
if export_figures
    exportgraphics(fig,fullfile(fig_dir,['optimization_' prefix '.pdf']),'ContentType','vector');
end
end

function matrix = history_matrix(runs,field)
lengths = cellfun(@(r)numel(r.(field)),runs);
matrix = NaN(numel(runs),max(lengths));
for j = 1:numel(runs)
    r = runs{j}; y = double(r.(field)(:)).';
    if isfield(r,'active_mask'), y(~logical(r.active_mask(:))) = NaN; end
    matrix(j,1:numel(y)) = y;
end
end

function matrix = diagnostic_matrix(runs,field)
matrix = NaN(numel(runs),0);
for j = 1:numel(runs)
    d = runs{j}.diagnostics;
    if ~isfield(d,field), continue; end
    idx = double(d.evaluation_iteration(:))+1;
    if isempty(idx), continue; end
    if size(matrix,2) < max(idx), matrix(:,end+1:max(idx)) = NaN; end
    matrix(j,idx) = double(d.(field)(:));
end
end

function drawn = draw_band(ax,matrix,color,label,style)
drawn = ~isempty(matrix) && any(isfinite(matrix),'all');
if ~drawn, return; end
matrix(~isfinite(matrix)) = NaN;
med = median(matrix,1,'omitnan');
lo = NaN(size(med)); hi = lo;
for k = 1:numel(med)
    values = sort(matrix(isfinite(matrix(:,k)),k));
    if isempty(values), continue; end
    % Linear interpolation of order statistics; no Statistics Toolbox required.
    lo(k) = percentile(values,.25); hi(k) = percentile(values,.75);
end
x = 0:numel(med)-1;
fill(ax,[x fliplr(x)],[lo fliplr(hi)],color,'FaceAlpha',.07,'EdgeColor','none','HandleVisibility','off');
plot(ax,x,med,'Color',color,'DisplayName',label,'LineStyle',style,'LineWidth',1.5,'HandleVisibility','off');
end

function value = percentile(sorted,p)
index = 1+(numel(sorted)-1)*p;
value = sorted(floor(index))+(index-floor(index))*(sorted(ceil(index))-sorted(floor(index)));
end

function decorate(ax,x,y)
grid(ax,'on'); box(ax,'on'); xlabel(ax,x,'Interpreter','none','Color','k'); ylabel(ax,y,'Interpreter','none','Color','k');
end

function style_legend(ax,labels,styles,available)
handles = gobjects(0); shown = {};
for j = find(available)
    handles(end+1) = plot(ax,NaN,NaN,'Color',[.2 .2 .2],'LineStyle',styles{j},'LineWidth',1.5); %#ok<AGROW>
    shown{end+1} = labels{j}; %#ok<AGROW>
end
if ~isempty(handles)
    legend(ax,handles,shown,'Location','best','Interpreter','none','FontSize',8,'TextColor','k','Color','w');
end
end

function colors = diagnostic_colors(n)
base = [0 .447 .698; .835 .369 0; 0 .620 .451; .800 .475 .655; ...
    .902 .624 0; .337 .706 .914; .600 .600 0; .494 .184 .556; .635 .078 .184];
if n <= size(base,1), colors = base(1:n,:); else, colors = [base;hsv(n-size(base,1))]; end
end
