
if ~exist('plotcolor')
    plotcolor = cell(1,0);
end
%figure()
mask = 1:length(rep_stats.mean_linstat.R_mean);
mask = mask(rep_stats.mean_linstat.evaluated > 0);

Rmean = -rep_stats.mean_linstat.R_mean(mask);
Rstd = rep_stats.std_linstat.R_mean(mask);
% R0mean = [0];
% R0std = [0];
% R0ind = [0];

R0mean = zeros(0,1);
R0std = zeros(0,1);
R0ind = zeros(0,1);

hold on
h = plot_confidence([R0ind; mask'], [R0mean; Rmean], [R0std; Rstd], plotcolor{:}, plotcolor{:});
scatter(mask', Rmean, 'k*');
if(rep_stats.mean_linstat.R_opt)
    plot([0; mask'], [rep_stats.mean_linstat.R_opt(1); rep_stats.mean_linstat.R_opt(mask)]);
end
xlabel('Episodes')
ylabel('Mean rewards')
set(gca,'fontsize',18)
