function [colors, markers] = record_length_method_styles(method_names)
% Fixed styles for record-length plots, independent of method order/subsets.
% lines(n) repeats its seven colors; each registered method has its own RGB.
keys=cellfun(@method_figure_label,cellstr(string(method_names)), ...
    'UniformOutput',false);
registry={'RGLS','WS-GNI','WS-GGI','GHAM-1-dH','GHAM-2-dH', ...
    'GHAM-2-H','GHAM-2-I','LHAM-1-TIK','LHAM-3-TIK','BI-HMC'};
palette=[ ...
    0.000 0.447 0.741;  % RGLS: blue
    0.850 0.325 0.098;  % WS-GNI: orange
    0.929 0.694 0.125;  % WS-GGI: gold
    0.494 0.184 0.556;  % GHAM-1-dH: purple
    0.466 0.674 0.188;  % GHAM-2-dH: green
    0.301 0.745 0.933;  % GHAM-2-H: sky blue
    0.635 0.078 0.184;  % GHAM-2-I: dark red
    0.050 0.240 0.480;  % LHAM-1-TIK: dark blue
    0.780 0.270 0.570;  % LHAM-3-TIK: pink
    0.120 0.120 0.120]; % BI-HMC: charcoal

% Allocate additional unique colors if a new method is compared later.
unknown=setdiff(keys,registry); candidate_index=0;
for k=1:numel(unknown)
    while true
        candidate_index=candidate_index+1;
        candidate=hsv2rgb([mod(candidate_index*.618033988749895,1) .72 .80]);
        if all(vecnorm(palette-candidate,2,2)>1e-6), break; end
    end
    registry{end+1}=unknown{k}; %#ok<AGROW>
    palette(end+1,:)=candidate; %#ok<AGROW>
end

% Use triangles for close-colored companions; keep this choice stable even
% when their comparison partners are absent from a particular method subset.
registry_markers=repmat({'o'},1,numel(registry));
for k=2:numel(registry)
    if any(vecnorm(palette(1:k-1,:)-palette(k,:),2,2)<.38)
        registry_markers{k}='^';
    end
end
registry_markers{strcmp(registry,'BI-HMC')}='^';
[found,index]=ismember(keys,registry);
assert(all(found),'record_length_method_styles:UnknownMethod','Missing method style.');
colors=palette(index,:); markers=registry_markers(index);
assert(size(unique(colors,'rows'),1)==numel(keys), ...
    'record_length_method_styles:DuplicateMethod', ...
    'Compared methods must have distinct labels and colors.');
end
