% [a, A, cov, rew, eta, theta] = reps_state(a, sigs, contextFunc, rewFunc, epsilon, samples, episodes, etatheta)
% a: initial policy mean (Gaussian policy)
% sigs: the standard deviations of the Gaussian policy
% contextFunc(N): callable function, samples N context variables (e.g. contextFunc(5) gives 5 context variables)
% rewFunc(context, policy parameter): callable function, gives the reward
% epsilon: the relative entropy upper bound
% samples: for each policy update the algorithm samples "samples" amount of samples :)
% episodes: number of policy updates
% etatheta: initial Lagrangian parameters initialize with some small random
% number, eta must be positive!!!

% sample call: simulateToyCannon([.3,0,0,0,0]', 3)

problem = ToyCannon0D1D2D;


episodes = 80;
epsilon = 1;
N = 50;
eta = 100;

a = mean(problem.theta_bounds,2); %policy mean
sigs = (problem.theta_bounds(:,2)-problem.theta_bounds(:,1))/2;
contextFunc = @(n)(samplerange(...
    [problem.st_bounds(:,1); problem.se_bounds(:,1)],...
    [problem.st_bounds(:,2); problem.se_bounds(2)], n));
rewFunc = @(s, theta)(-problem.sim_eval_func([s, theta])); %keep bounds
samples = N;
etatheta = 0.1*rand(1,3+size(problem.se_bounds,1)+size(problem.st_bounds,1));

[a, A, cov, rew, eta, theta, hist] = reps_state(a, sigs, contextFunc, rewFunc, epsilon, samples, episodes, etatheta)



% 
% options = optimset('Algorithm','active-set');
% options = optimset(options, 'GradObj','on');
% options = optimset(options, 'Display', 'off');
% 
% for i = 1:episodes
%     S = [];
%     rews = [];
%     for j = 1:N
%         par_loc = mvnrnd(mean_par, cov_par, 1)';
%         S(j, :) = par_loc';
%         f = @(z) activation_func(z, centers, sig)' * par_loc / sum(activation_func(z, centers, sig));
%         
%         dmp_diff_loc = @(t, x) dmp_diff(t, x, f, g, tau, alpha_x, beta_x, alpha_z);
%         [t, y] = ode45(dmp_diff_loc, tm, [0 3 1]);
%         
%         for k = 1:tlen
%             acc(k) = accFromDMP(y(k, 1), y(k, 2), y(k, 3), f);
%         end
%         tq = torque(y(:, 1), acc(:));
%         
%         savem.pos = [savem.pos, y(:, 1)];
%         savem.vel = [savem.vel, y(:, 2)];
%         savem.acc = [savem.acc, acc(:)];
%         savem.tq  = [savem.tq, tq(:)];
%         
%         rews(j) = rew(tq, y);
%     end
%     
%     r = rews(:);
%     q = ones(N, 1)/N;
%     
%     objfun = @(eta) reps_dual(eta, r, q, epsilon);
%     logeta = minimize(log(eta), objfun, -100);
%     if ~isnan(logeta)
%         eta = exp(logeta);
%     end
%         
%     
%     p = exp(r/eta)/sum(exp(r/eta));
%     % 	Dkl = sum(p.*log(p./q));
%     % 	iDkl = sum(q.*log(q./p));
%     mn = S'*p;
%     cn = (S-repmat(mn', size(S, 1), 1))'*((S-repmat(mn', size(S, 1), 1)).*repmat(p, 1, length(centers)));
%     eig(cn)
%     
%     mean_par = mn;
%     cov_par = cn;
%     
%     % 	rexp = r(:)'*p;
%     % 	rvar = (r(:)-rexp)'*((r(:)-rexp).*p);
%     figure(1),clf
%     subplot(2,2,1)
%     shadedErrorBar(tm, mean(savem.pos, 2), 2*std(savem.pos')', 'b-', 1); hold on
% plot(target(1,1)*dt, target(1,2), 'ro')
% plot(target(2,1)*dt, target(2,2), 'ro')
% plot(target(3,1)*dt, target(3,2), 'ro')
% plot(tend, 1, 'ro')
%     ylabel('Pos')
%     xlabel('time [sec]')
%     subplot(2,2,2)
%     shadedErrorBar(tm, mean(savem.vel, 2), 2*std(savem.vel')', 'r-', 1); hold on
%     ylabel('Vel')
%     xlabel('time [sec]')
%     subplot(2,2,3)
%     % shadedErrorBar(tm, mean(savem.acc, 2), 2*std(savem.acc')', 'g-', 1); hold on
%     shadedErrorBar(tm, mean(savem.tq, 2), 2*std(savem.tq')', 'k-', 1); hold on
%     ylabel('Torque')
%     xlabel('time [sec]')
%     subplot(2,2,4)
%     h = histfit(rews, 10);
%     set(h(1),'FaceColor','w','EdgeColor','k')
%     ylabel('Reward dist.')
%     drawnow
%     pause(.5)
% end
