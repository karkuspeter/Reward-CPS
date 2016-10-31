%% show specific result
figure
indices = 1:numel(hp_list); %[4 5 9 10];%1:numel(hp_list);
%indices = 1:4;
labels = {};
plots = {};
for ind = indices
    rep_stats = h_stats{ind};
    linstat_vec = h_linstats{ind};
    params = h_params{ind};
    labels = [labels, {num2str(ind)}];
    params
    ShowRepeatResults2()
    plots = [plots, {h}];
end
legend([plots{:}], labels);