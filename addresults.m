%% load two files
clear all;
%load('../../results/10-16-01-29-38-se-alg4-16/hyper-16-10-2016-00-34.mat');
load('NIPS/st-alg1-16.mat');
ShowTuneResults;
RES = load('NIPS/alg4-48-comb.mat');

%% combine two param settings

hp_list = cat(1, hp_list, RES.hp_list);
h_stats = cat(2, h_stats, RES.h_stats);
h_linstats = cat(2, h_linstats, RES.h_linstats);
h_params = cat(2, h_params, RES.h_params);

ShowTuneResults;

%% save

save('combined.mat');


%% combine alg4 results
clear all;
load('NIPS/se-4-300/hyper-16-10-2016-14-17.mat')
RES = load('NIPS/se-4-300/hyper-16-10-2016-14-52.mat');

save('NIPS/se-4-48.mat');

clear all;
load('NIPS/st-4-300/hyper-16-10-2016-13-16.mat')
RES = load('NIPS/st-4-300/hyper-16-10-2016-14-02.mat');

save('NIPS/st-4-48.mat');

clear all;
load('NIPS/st-4-48.mat')
RES = load('NIPS/se-4-48.mat');

hp_list = cat(1, hp_list, RES.hp_list);
h_stats = cat(2, h_stats, RES.h_stats);
h_linstats = cat(2, h_linstats, RES.h_linstats);
h_params = cat(2, h_params, RES.h_params);

save('NIPS/alg4-48-comb.mat');

%% combine repeats
ind1 = 1;
ind2 = 1;

if ~isequal(h_params{ind1},RES.h_params{ind2})
    disp(h_params{ind1} )
    disp(RES.h_params{ind2} )
    error('');
end
h_linstats{ind1} = cat(1, h_linstats{ind1}, RES.h_linstats{ind2});

h_stats{ind1}.Rcumm_mean = 0; 
h_stats{ind1}.Rcumm_std = 0; 
h_stats{ind1}.Rand_seed = cat(1, h_stats{ind1}.Rand_seed, RES.h_stats{ind2}.Rand_seed);
h_stats{ind1}.stat_vec = cat(1, h_stats{ind1}.stat_vec, RES.h_stats{ind2}.stat_vec);

linstat_vec = h_linstats{ind1};
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
    end
    if ~isstruct(mean_linstat.(fields{i}))
        std_linstat.(fields{i}) = std(mean_linstat.(fields{i}), 0, dim);
        mean_linstat.(fields{i}) = mean(mean_linstat.(fields{i}), dim);
    end
end
h_stats{ind1}.mean_linstat = mean_linstat;
h_stats{ind1}.std_linstat = std_linstat;
disp('done')
disp(size(h_linstats{ind1},1))
%% save

save('combined.mat');


%% pretty plot
colorOrder = get(gca, 'ColorOrder');
colors = [ colorOrder(2,:); colorOrder(1,:); colorOrder(3,:) ];

figure
indices = 1:numel(hp_list); %[4 5 9 10];%1:numel(hp_list);
indices = [3, 2, 1];
labels = {};
plots = {};
for ii = 1:length(indices)
    ind=indices(ii);
    rep_stats = h_stats{ind};
    linstat_vec = h_linstats{ind};
    params = h_params{ind};
    labels = [labels, {num2str(ind)}];
    params
    plotcolor = {colors(ind,:)};
    ShowRepeatResults2()
    plots = [plots, {h}];
end
legend([plots{:}], labels);set(gca,'color','none')
set(gca,'fontsize',16);

ylim([-6; 0.5])
 hleg = legend([plots{[2, 1, 3]}], { ...
     sprintf('FC-BPS (ours)'), ...
     sprintf('Active FC-BPS (ours)'), ...
     sprintf('BO-CPS (baseline)'), ...
     }, 'Location', 'southeast');
set(hleg, 'color', 'none');

%legend(gca,'off');
%%
export_fig env2.png -transparent -m5

%% plot problem
t = ToyCannonBase3D;
t.Randomise();

%% sdf
s1 = 5; s2 = 0;
ang_h = atan2(s2, s1);
ang_h(ang_h<0) = ang_h(ang_h<0) + 2*pi;

[x1, x3] = ndgrid(linspace(t.theta_bounds(1,1), t.theta_bounds(1,2), 100),...
            linspace(t.theta_bounds(3,1), t.theta_bounds(3,2), 100));

vec = ones(size(x1(:)));
val_full = t.toycannon.Simulate(s1*vec, s2*vec, x1(:), ang_h*vec, x3(:), 0);

[r_opt, ind] = min(val_full);

th = [x1(ind), ang_h, x3(ind)]

t.toycannon.PrintOn = true;
[r, result] = t.toycannon.Simulate( s1, s2, th(1), th(2), th(3), 0)
t.toycannon.PrintOn = false;
xlim([-8; 8]); ylim([-8; 8]);
set(gca,'color','none')
set(gca,'fontsize',16);

%% combine with only some trials

ind1 = 1;
ind2 = 1;
sel = 1:2;

if ~isequal(h_params{ind1},RES.h_params{ind2})
    disp(h_params{ind1} )
    disp(RES.h_params{ind2} )
    error('');
end
h_linstats{ind1} = cat(1, h_linstats{ind1}, RES.h_linstats{ind2}(sel,:));

h_stats{ind1}.Rcumm_mean = 0; 
h_stats{ind1}.Rcumm_std = 0; 
h_stats{ind1}.Rand_seed = cat(1, h_stats{ind1}.Rand_seed, RES.h_stats{ind2}.Rand_seed(sel,:));
h_stats{ind1}.stat_vec = cat(1, h_stats{ind1}.stat_vec, RES.h_stats{ind2}.stat_vec(sel,:));

linstat_vec = h_linstats{ind1};
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
    end
    if ~isstruct(mean_linstat.(fields{i}))
        std_linstat.(fields{i}) = std(mean_linstat.(fields{i}), 0, dim);
        mean_linstat.(fields{i}) = mean(mean_linstat.(fields{i}), dim);
    end
end
h_stats{ind1}.mean_linstat = mean_linstat;
h_stats{ind1}.std_linstat = std_linstat;
disp('done')
disp(size(h_linstats{ind1},1))