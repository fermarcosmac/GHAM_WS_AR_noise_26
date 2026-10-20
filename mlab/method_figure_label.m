function label = method_figure_label(method_name)
% Figure acronyms only; optimizer identifiers in configs/results stay unchanged.
label = char(method_name);
label = regexprep(label, 'GGOTHAM|GGHAM', 'GHAM');
label = regexprep(label, 'LGOTHAM|LGHAM', 'LHAM');
if contains(label,'HAM'), label = regexprep(label,'^WS-',''); end
end
