function reporter = study_progress(label, methods, n_runs, lengths, n_resolutions, settings)
% Default study logging. ETA uses observed seconds/sample separately per method.
enabled = field_or(settings,'progress',true);
interval = field_or(settings,'progress_interval_seconds',10);
assert(isscalar(interval) && isfinite(interval) && interval>0, ...
    'progress_interval_seconds must be positive and finite.');
methods = cellstr(string(methods));
total = numel(methods)*n_runs*numel(lengths)*n_resolutions;
remaining = repmat(n_runs*sum(lengths)*n_resolutions,numel(methods),1);
samples = zeros(numel(methods),1); seconds = samples;
completed = 0; started = tic; last_report = 0;
active_method = []; active_length = 0; method_started = 0;
reporter = struct('enabled',enabled,'callback',@update,'stage',@stage,'finish',@finish);
stage(sprintf('Starting %d repetitions, %d method fits; ETA warms up after the first fit of each method.',n_runs,total));

    function stage(message, force)
        if ~enabled, return; end
        if nargin<2, force=true; end
        elapsed = toc(started);
        if force || elapsed-last_report>=interval
            fprintf('[%s] %s | elapsed %s\n',label,message,duration(elapsed));
            last_report = elapsed;
        end
    end

    function update(event)
        if ~enabled, return; end
        if strcmp(event.phase,'stage'), stage(event.message); return; end
        elapsed = toc(started);
        m = find(strcmp(methods,event.method),1);
        switch event.phase
            case 'method_start'
                active_method = m; active_length = event.record_length; method_started = elapsed;
                detail = sprintf('starting; update budget %d',event.max_updates);
            case 'iteration'
                if elapsed-last_report<interval, return; end
                detail = sprintf('updates %d/%d',event.iteration,event.max_updates);
            case 'method_end'
                fit_time = elapsed-method_started;
                seconds(m) = seconds(m)+fit_time;
                samples(m) = samples(m)+event.record_length;
                remaining(m) = remaining(m)-event.record_length;
                completed = completed+1; active_method = [];
                detail = sprintf('%s; %d accepted updates; fit time %s', ...
                    event.status,event.iteration,duration(fit_time));
            otherwise
                error('Unknown progress event: %s',event.phase);
        end
        if contains(label,'initialization')
            context = sprintf('initialization %d/%d',event.metadata.mc_index,n_runs);
        else
            context = sprintf('recursive_PL; N=%d; realization %d/%d', ...
                event.record_length,event.metadata.mc_index,n_runs);
        end
        estimate = 'warming up';
        if all(samples>0)
            eta = sum(remaining.*seconds./samples);
            if ~isempty(active_method)
                eta = eta-min(elapsed-method_started, ...
                    active_length*seconds(active_method)/samples(active_method));
            end
            estimate = ['~' duration(eta)];
        end
        fprintf('[%s] %d/%d fits completed (%.1f%%) | %s | %s: %s | elapsed %s | ETA %s\n', ...
            label,completed,total,100*completed/total,context,event.method,detail,duration(elapsed),estimate);
        last_report = elapsed;
    end

    function finish(path)
        stage(sprintf('Finished %d/%d method fits; saved %s',completed,total,path));
    end
end

function text = duration(seconds)
seconds = round(max(0,seconds));
text = sprintf('%02d:%02d:%02d',floor(seconds/3600),floor(mod(seconds,3600)/60),mod(seconds,60));
end

function value = field_or(s,key,default)
if isfield(s,key), value=s.(key); else, value=default; end
end
