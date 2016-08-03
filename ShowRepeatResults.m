
%figure()
hold on
plot_confidence((1:size(rep_stats.mean_linstat.R_mean,1))', rep_stats.mean_linstat.R_mean, rep_stats.std_linstat.R_mean);
hold on
plot((1:size(rep_stats.mean_linstat.R_mean,1))', rep_stats.mean_linstat.R_opt);
xlabel('Iteration')
ylabel('R averaged over all contexts')