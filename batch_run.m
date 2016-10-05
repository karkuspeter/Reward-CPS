init

%hyper_params = struct('Algorithm', [1,2]);
hyper_params = struct(...
    'problem', '{ToyCannon1D0D8D}', ...
    'kappa', '{5}', ...
    ...%'LearnHypers', '{true}', ...
    'Ntrial_st', '{100}', ...  %representers for st space, can be number or vector
    'Algorithm', '{1, 2, 4}' ... %);
    );
common_params = struct('output_off', 1, ...
    'Niter', 100, ...
    'InitialSamples', 14, ...
    'Neval', [50 20 20 20 20 20 20 20 20 20 20 20 20], ...
    'OptimisticMean', -1, ... %lowest possible value (will shift y values)
    'EvalModulo', 5);
repeat_setting = 100;

% create a list of all permutations     
hp_list = [common_params]; % start with params for all executions
param_names = fieldnames(hyper_params);
for param_name = param_names(:)'
    hp_list2 = [];
    for i_tuner = 1:numel(hp_list)
        list_el = hp_list(i_tuner);
        param_values = eval(hyper_params.(param_name{:}));
        for j = 1:length(param_values)
            param_val = param_values(j);
            list_el.(param_name{:}) = param_val{:};
            hp_list2 = [hp_list2; list_el];
        end
    end
    hp_list = hp_list2;
end

%new figure for plotting during repeat
figure
hold on

% execute each
if (true)
    h_stats = {};
    h_linstats = {};
    h_params = {};
    i_tuner_start = 1;
else
    i_tuner_start = 3;
end

seed = rng();
for i_tuner = i_tuner_start:numel(hp_list)  %dont use i, its being overwritten inside
    list_el = hp_list(i_tuner);
    rng(seed);
    
    repeats = repeat_setting;
    keep_prev = 0;
        
    run_struct = list_el

    %default
    spider_func = @MyEntropySearch;%@RBOCPS;
    show_func = @ShowRepeatResults;
    reward_name = 'R_mean';
    
    if isfield(run_struct, 'spider_func')
        %str = strcat(run_struct.spider_func,'(varagin)');
        spider_func = run_struct.spider_func;
        run_struct = rmfield(run_struct, 'spider_func');
    end
    % manually compute iterations to be always the same
    
    no_params = 1;
    repeat;
    
    % save results
    h_stats{i_tuner} = rep_stats;
    h_linstats{i_tuner} = linstat_vec;
    h_params{i_tuner} = params;
    
    save('results/hyperparam_temp.mat');
    drawnow
end

% make summary
res_list = hp_list;
% for i=1:numel(res_list)
%     res_list(i).total_samples = h_stats(i).mean_linstat.total_samples;
%     res_list(i).Rcumm_mean = h_stats(i).Rcumm_mean;
%     res_list(i).Rcumm_std = h_stats(i).Rcumm_std;
% end

%save
save(strcat('results/hyper-', datestr(now,'dd-mm-yyyy-HH-MM'), '.mat'));

% show specific result
figure

%indices = 1:numel(hp_list); %[4 5 9 10];%1:numel(hp_list);
indices = 1:4;
labels = {};
plots = {};
for ind = indices
    rep_stats = h_stats{ind};
    linstat_vec = h_linstats{ind};
    params = h_params{ind};
    labels = [labels, {num2str(ind)}];
    params
    show_func()
    plots = [plots, {h}];
end
legend([plots{:}], labels);

h_cumm_mean = [];
h_cumm_std = [];
for ind = 1:length(hp_list)
    h_cumm_mean(ind) = h_stats{ind}(1,1).Rcumm_mean;
    h_cumm_std(ind) = h_stats{ind}(1,1).Rcumm_std;
end
h_eval_matrix = [1:length(hp_list); h_cumm_mean; h_cumm_std];

[maxval, maxind] = max(h_cumm_mean)
best_params = h_params{maxind}

no_params = 0;