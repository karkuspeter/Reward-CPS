if ~exist('no_params') || no_params == 0
    repeats = 2;
    keep_prev = 0;
    spider_func = @RBOCPS;
    show_func = @ShowRepeatResults;
    reward_name = 'R_mean';
    run_struct = struct('output_off',1,'method',2);
end

if (~keep_prev) || ~exist(stat_vec)
    stat_vec = [];
    linstat_vec = [];
    cumm_rew_vec = [];
    seed_vec = [];
end

for i=1:repeats
    disp(sprintf('%d / %d', i, repeats));
    seed_vec = [seed_vec; rng()];
    
    [stat, linstat, params] = spider_func(run_struct);
    cumm_rew = mean(linstat.(reward_name));
rep_stats.Rand_seed = seed_vec;
    stat_vec = [stat_vec; stat];
    linstat_vec = [linstat_vec; linstat];
    % params should be same, so we dont store it in vector
    cumm_rew_vec(:,:,end+1) = cumm_rew;
end

rep_stats.Rcumm_mean = mean(cumm_rew_vec,3);
rep_stats.Rcumm_std = std(cumm_rew_vec,0,3);


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
    std_linstat.(fields{i}) = std(mean_linstat.(fields{i}), 0, dim);
    mean_linstat.(fields{i}) = mean(mean_linstat.(fields{i}), dim);
    %mean_linstat.(fields{i}) = mean_linstat.(fields{i}) ./ length(linstat_vec);
end
rep_stats.mean_linstat = mean_linstat;
rep_stats.std_linstat = std_linstat;


show_func()