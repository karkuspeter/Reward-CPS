
%figure()
mask = 1:length(rep_stats.mean_linstat.R_mean);
mask = mask(rep_stats.mean_linstat.evaluated > 0);

hold on
plot_confidence(mask', rep_stats.mean_linstat.R_mean(mask), rep_stats.std_linstat.R_mean(mask));
plot(mask', rep_stats.mean_linstat.R_opt(mask));
xlabel('Iteration')
ylabel('R averaged over all contexts')