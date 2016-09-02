% run init.m before
addpath ./ACES

% set up prior belief
N               = 2; % number of input dimensions
in.covfunc      = {@covSEard};       % GP kernel
in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard, covRQard, respectively.

% should the hyperparameters be learned, too?
in.LearnHypers  = false; % yes.
in.HyperPrior   = @SEGammaHyperPosterior;

in.MaxEval      = 10;    % Horizon (number of evaluations allowed)

% constraints defining search space:
% objective function:
%in.xmin         = [-0.2, -0.2]; % lower bounds of rectangular search domain
%in.xmax         = [0.2, 0.2]; % upper bounds of rectangular search domain
%in.f            = @(x) PhysicalExperiment(x); % handle to objective function
%R_func = @(R, theta)(R + min(0,sigmf(mean(abs(theta),2), [10 0.3])*1-0.5));
%in.f = @(x)(arrayfun(@(a,b)(R_func(0, abs(a)+abs(b)).*4 + 1), x(:,1), x(:,2)));

problem = ToyCannon1D2D;
in.xmin = [problem.st_bounds(1,1)/10 problem.theta_bounds(1,1)' ];
in.xmax = [problem.st_bounds(1,2)/10 problem.theta_bounds(1,2)' ]
in.f = @(x)(arrayfun(@(a,b)(-problem.sim_func(a*10, [b 1])/5-0.4), x(:,1), x(:,2)));
hyp.cov         = log([0.3^2; 0.45^2; 1]); % hyperparameters for the kernel
%TODO these are not normalized!
hyp.lik         = log([3e-3]); % noise level on signals (log(standard deviation));
in.hyp          = hyp;  % hyperparameters, with fields .lik (noise level) and .cov (kernel hyperparameters), see documentation of the kernel functions for details.
in.MapFunc = @(GP,st)(MapFunc(GP, st, @(s, x, out)(problem.r_func(s*10, [x 1], out))));

% in.xmin = problem.theta_bounds(:,1)';
% in.xmax = problem.theta_bounds(:,2)';
% in.f = @(x)(arrayfun(@(a,b)(-problem.sim_func(4, [a b])), x(:,1), x(:,2)));

rng(1, 'twister');

% Initial samples
initx = [];
inity= [];
for i=1:5
    initx(i, :) = ((in.xmax'-in.xmin').*rand(size(in.xmin',1),1) + in.xmin');
end
initx = [initx; 0.7065  1.3708];
in.x = initx;
inity = in.f(initx);
in.y = inity;

%%replace for reproducing something

 in.x=[
    0.5170    0.9902;
    0.1001    0.4214;
    0.2468    0.1357;
    0.2863    0.4802;
    0.4968    0.7432;
    0.7065    1.3708;
    0.2667    0.0856; ];
in.y = [   
    -0.1540;
    0.2798;
   -0.9265;
    0.1075;
    0.0265;
   -0.3270;
   -0.8067];

%%
[xx, xy] = meshgrid([in.xmin(1):0.05:in.xmax(1)], [in.xmin(2):0.01:in.xmax(2)]);
%mesh(xx, xy, arrayfun(@(a,b)(in.f([a b])), xx, xy))

%% run optimization
tic
result = MyEntropySearch(in) % the output is a struct which contains GP datasets, which can be evaluated with the gpml toolbox.
toc

%% Visualization of results
perf = sum(result.val_vec, 1);
figure
plot(perf);
hold on
plot(ones(size(perf))*sum(result.val_opt));

return

% just brief intro...
figure
mesh(xx, xy, arrayfun(@(a,b)(in.f([a b])), xx, xy))

hold on;
scatter3(result.FunEst(:,1), result.FunEst(:,2), in.f(result.FunEst), 'y*');
scatter3(result.FunEst(end,1), result.FunEst(end,2), in.f(result.FunEst(end,:)), 'gx');
scatter3(result.GP.x(:,1), result.GP.x(:,2), in.f(result.GP.x), 'ro');
hold off;
datarange = zlim();

% All 5 models visualization
figure
for i = 1:in.MaxEval
  if i == in.MaxEval
      figure
  else
    subplot(1,in.MaxEval-1,i);
  end
  [my_m my_s2] = gp(result.GPs{i}.hyp, [], [], result.GPs{i}.covfunc, result.GPs{i}.likfunc, result.FunEst(1:i,:), in.f(result.FunEst(1:i,:)), [xx(:) xy(:)]);
  %plot_confidence(x, my_m, my_s2);
  mesh(xx,xy, reshape(my_m, size(xx)));
  hold on
    scatter3(result.GP.x(1:i,1), result.GP.x(1:i,2), in.f(result.GP.x(1:i,:)), 'ro');
    scatter3(result.FunEst(i,1), result.FunEst(i,2), in.f(result.FunEst(i,:)), 'y*');    
    hold off
  %plotErr1(x', my_m, my_s2, result.FunEst(1:i,:)', in.f(result.FunEst(1:i,:)'));
  %axis([in.xmin(1) in.xmax(1) in.xmin(2) in.xmax(2) -0.5 -0.2]);
  zlim(datarange);
end;

[result.FunEst, in.f(result.FunEst)]

%GP = result.GP;
%[ymu, ysigma] = gp(GP.hyp, 'infExact', 

