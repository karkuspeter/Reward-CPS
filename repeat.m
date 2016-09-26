if ~exist('no_params') || no_params == 0
    repeats = 6;
    keep_prev = 0;
    spider_func = @RBOCPS; %@MyEntropySearch; %@RBOCPS;
    show_func = @ShowRepeatResults;
    reward_name = 'R_mean';
    run_struct = struct('output_off',1,'method',2);
    NeedNewFigure = true;
end

if (~keep_prev) || ~exist(stat_vec)
    stat_vec = {};
    linstat_vec = {};
    cumm_rew_vec = [];
    seed_vec = {};
end
params_vec = {};

reps = size(stat_vec,1)+1:size(stat_vec,1)+repeats;
parfor i=reps
    fprintf('%d / %d', i, repeats);
    % random seed is supposed to be different in each worker, just save it for reproducability
    seed_vec{i} = rng();
    
    [stat, linstat, params] = spider_func(run_struct);
    cumm_rew = mean(linstat.(reward_name));

    stat_vec{i} = stat;
    linstat_vec{i} = linstat;
    params_vec{i} = params;
    % params should be same, so we dont store it in vector
    cumm_rew_vec(:,:,i) = cumm_rew;
    
    %if (linstat.R_mean(end) < 2.5)
    %    disp('Low performance');
    %end
end
stat_vec = [stat_vec{:}]';
linstat_vec = [linstat_vec{:}]';
seed_vec = [seed_vec{:}]';
params = params_vec{1}; %to ensure params is propagated outside of parfor

rep_stats.Rcumm_mean = mean(cumm_rew_vec,3);
rep_stats.Rcumm_std = std(cumm_rew_vec,0,3);
rep_stats.Rand_seed = seed_vec;
rep_stats.stat_vec = stat_vec;

% compue mean and std of everything
mean_linstat = linstat_vec(1);
std_linstat = linstat_vec(1);
fields = fieldnames(mean_linstat);
for i=1:numel(fields)
    % mean_linstat.(fields{i}) = linstat_vec(1).(fields{i}); %alread equals
    % to this
    dim = ndims(mean_linstat.(fields{i}))+1;
    
    for j=2:length(linstat_vec)
        mean_linstat.(fields{i}) = cat(dim, mean_linstat.(fields{i}), linstat_vec(j).(fields{i}));
        %mean_linstat.(fields{i}) = mean_linstat.(fields{i}) + linstat_vec(j).(fields{i});
    end
    if ~isstruct(mean_linstat.(fields{i}))
        std_linstat.(fields{i}) = std(mean_linstat.(fields{i}), 0, dim);
        mean_linstat.(fields{i}) = mean(mean_linstat.(fields{i}), dim);
    end
    %mean_linstat.(fields{i}) = mean_linstat.(fields{i}) ./ length(linstat_vec);
end
rep_stats.mean_linstat = mean_linstat;
rep_stats.std_linstat = std_linstat;
   
if exist('NeedNewFigure') && NeedNewFigure 
    figure
end
show_func()