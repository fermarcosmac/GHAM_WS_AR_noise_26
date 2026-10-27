function datasets = run_parameter_error_win_analysis(options)
% Regenerate both studies' exact paired tests for both benchmark systems.
% run_parameter_error_win_analysis(struct('fig_dir',custom_dir))
if nargin<1, options=struct(); end
experiments={'example_1','example_CSTR'};
visibility=get(groot,'defaultFigureVisible');
restore=onCleanup(@()set(groot,'defaultFigureVisible',visibility)); %#ok<NASGU>
set(groot,'defaultFigureVisible','off');
datasets=struct([]);
for k=1:numel(experiments)
    options.experiment_name=experiments{k};
    options.mode='MONTECARLO'; options.include_initialization=true;
    options.statistics_only=true;
    loaded=plot_comparison_results(options);
    if isempty(datasets), datasets=loaded; else, datasets=[datasets loaded]; end %#ok<AGROW>
    close all;
end
end
