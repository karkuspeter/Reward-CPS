%hyper_params = struct('Algorithm', [1,2]);
isdirect = true;
setting = struct('problem', ToyBall, 'Algorithm', 4, 'Niter', 100, 'EvalModulo', 20);  
direct_settings = setting;
setting = struct('problem', ToyBallSE, 'Algorithm', 4, 'Niter', 400, 'EvalModulo', 50);  
direct_settings = [direct_settings; setting];
%    -0.2681    -0.2388    0.2597    0.8337    0.8468    0.5859

hyper_params = struct(...
    'problem', '{ToyBall, ToyBallSE}', ...
    'kappa', '{1}', ...
    ...%'LearnHypers', '{true}', ...
    'Ntrial_st', '{1}', ...  %representers for st space, can be number or vector
    'Algorithm', '{4}' ... %);
    );

common_params = struct('output_off', 0, ...
    'RandomiseProblem', true, ...
    'ReturnOptimal', 0, ... %computes optimal values and put in return struct
    'InitialSamples', 19, ...
    'Neval', [8 8 50 360 50 20 20 20 20 20 20 20], ...
    'OptimisticMean', 0.5);
repeat_setting = 2;
fix_seeds = true;

% create a list of all permutations     
hp_list = [common_params]; % start with params for all executions
if isdirect
    hp_list = repmat(hp_list, size(direct_settings, 1), 1);
    param_names = [fieldnames(hp_list); fieldnames(direct_settings)];
	hp_list = cell2struct([struct2cell(hp_list); struct2cell(direct_settings)], param_names, 1);
else
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
    % create seed list
    if fix_seeds
        fixed_seeds = cell(repeat_setting,1);
        for ind = 1:repeat_setting
            rng('shuffle');
            fixed_seeds{ind} = rng();
        end
    else
        fixed_seeds = {};
    end
else
    i_tuner_start = 3;
end

for i_tuner = i_tuner_start:numel(hp_list)  %dont use i, its being overwritten inside
    list_el = hp_list(i_tuner);
    
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

%% show specific result
figure
indices = 1:numel(hp_list); %[4 5 9 10];%1:numel(hp_list);
%indices = [6:9];
labels = {};
plots = {};
for ind = indices
    rep_stats = h_stats{ind};
    linstat_vec = h_linstats{ind};
    params = h_params{ind};
    labels = [labels, {num2str(ind)}];
    params
    ShowRepeatResults()
    plots = [plots, {h}];
end
legend([plots{:}], labels);

%% evaluate successes
tresh = 0.9;
conv_vec = zeros(4,numel(hp_list));  % index, mean iteration to reach tresh, std iteration, fail count
for ind = 1:numel(hp_list)
    linstat_vec = h_linstats{ind};
    temp_vec = [];
    for ind_rep=1:repeat_setting
        
        temp = find(linstat_vec(ind_rep).R_mean > tresh, 1);
        if isempty(temp)
            % failed, never reach treshold reward
            conv_vec(4, ind) = conv_vec(4, ind) + 1;
        else
            temp_vec = [temp_vec; temp];
        end
    end
    conv_vec(1, ind) = ind;
    conv_vec(2, ind) = mean(temp_vec);
    conv_vec(3, ind) = std(temp_vec);
end
conv_vec

%%
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

%% alternative evaluation when not all iterations are evaluated
h_cumm_mean = [];
h_cumm_std = [];
for ind_p = 1:length(hp_list)
    R_vec_rel = [];
    for ind_rep = 1:repeat_setting
        R_vec = h_linstats{ind_p}(ind_rep).R_mean;
        eval_vec = h_linstats{ind_p}(ind_rep).evaluated;
        R_vec_rel = [R_vec_rel; R_vec(eval_vec > 0)];
    end
    h_cumm_mean(ind_p) = mean(R_vec_rel);
    h_cumm_std(ind_p) = std(R_vec_rel);
end
h_eval_matrix = [1:length(hp_list); h_cumm_mean; h_cumm_std];
[maxval, maxind] = min(h_cumm_mean)
best_params = h_params{maxind}
[~, idsort] = sort(h_cumm_mean);
h_eval_matrix_sorted = h_eval_matrix(:,idsort);
