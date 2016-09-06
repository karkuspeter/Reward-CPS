

%figure()
mask = 1:length(rep_stats.mean_linstat.R_mean);
mask = mask(rep_stats.mean_linstat.evaluated > 0);

hold on
h = plot_confidence([0; mask'], [0; rep_stats.mean_linstat.R_mean(mask)], [0; rep_stats.std_linstat.R_mean(mask)]);
scatter(mask', rep_stats.mean_linstat.R_mean(mask), 'r*');
if(rep_stats.mean_linstat.R_opt)
    plot([0; mask'], [rep_stats.mean_linstat.R_opt(1); rep_stats.mean_linstat.R_opt(mask)]);
end
xlabel('Iteration')
ylabel('R averaged over all contexts')

