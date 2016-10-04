function [ stats, linstat, params ] = creps(input_params)

%TODO: it will perform bad with noisy observations
%add Ntheta / Nsamples

% for st context generate M samples and reweight

params = struct(...
    'problem', ToyCannon1D0D2D, ...
    'Neval', [20, 20, 20], ... %evaluation points over contexts
    'Algorithm', 3, ...   % 1 FCREPS, 2 CREPS, 3 FCREPS with myreps
    'Nart', 1, ...     %number of artificial st samples for FCREPS
    'Niter', 1100, ...  %number of interactions with world. Episodes = Niter/Nsamples;
    'Nsamples', 10, ... %number of samples obtained from system at each episode
    'epsilon', 1, ... %epsilon for REPS (entropy bound, should be around 1)
	'eta', 0.001, ...100, ... %eta for REPS
    'mu_init', [], ... %[theta_dim, 1]; default mean(problem.theta_bounds,2)
    'sigma_init', [], ... %[theta_dim, 1]; default sigs = (problem.theta_bounds(:,2)-problem.theta_bounds(:,1))/2
    'InitialSamples', 0, ...  %minimum 1
    'EvalModulo', 1, ... % modulo in terms of episodes, not samples like in Bayesian
    'ReturnOptimal', 1, ... %computes optimal values and put in return struct
    'output_off', 1);

if (exist('input_params'))
    params = ProcessParams(params, input_params);
end

problem = params.problem;
params.problem = struct; %clear so dont have to pass in to parfor later

params.xmin = [problem.st_bounds(:,1)' problem.se_bounds(:,1)' problem.theta_bounds(:,1)'];
params.xmax = [problem.st_bounds(:,2)' problem.se_bounds(:,2)' problem.theta_bounds(:,2)'];
params.D = size(params.xmax,2); % dimensionality of inputs (search domain)

D = params.D;
st_dim = size(problem.st_bounds,1);
se_dim = size(problem.se_bounds,1);
params.st_dim = st_dim;
params.se_dim = se_dim;
s_dim = params.st_dim + params.se_dim;
th_dim = D-s_dim;

stats = struct('last_R_mean', 0);
linstat = struct('R_mean', zeros(params.InitialSamples,1), ...
                 ...%'st', zeros(size(GP.x,1),1), 'se', GP.x(:,sei-st_dim), 'theta', GP.x(:,thi-st_dim), ...
                 'theta_s', zeros(params.InitialSamples, prod(params.Neval(1:s_dim))*th_dim), ...
                 'R_s', zeros(params.InitialSamples, prod(params.Neval(1:s_dim))), ...
                 'R_opt', [], ...
                 'evaluated', zeros(params.InitialSamples,1) ...
                 );
%TODO
% uses sim_eval_func for training
             
%% REPS formulation
episodes = params.Niter/params.Nsamples;

a = params.mu_init;
sigs = params.sigma_init;
if isempty(a)
    a = mean(problem.theta_bounds,2); %policy mean
end
if isempty(sigs)
    sigs = (problem.theta_bounds(:,2)-problem.theta_bounds(:,1))/2;
end
stFunc = @(n)(samplerange(...
    problem.st_bounds(:,1), problem.st_bounds(:,2), n));
seFunc = @(n)(samplerange(...
    problem.se_bounds(:,1), problem.se_bounds(:,2), n));
contextFunc = @(n)([stFunc(n); seFunc(n)]);

simFunc = @(s, theta)(problem.sim_eval_func([s, theta])); %keep bounds
rewFunc = @(s, theta, obs)(problem.r_func(s, theta, obs));
etatheta = [params.eta, 0.1*rand(1,2+size(problem.se_bounds,1)+size(problem.st_bounds,1))];

% execute REPS
if (params.Algorithm == 1 || params.Algorithm == 3)
    [a, A, cov, rew, eta, theta, hist] = fact_reps_state(a, sigs, ...
           problem.st_bounds', stFunc, seFunc, rewFunc, simFunc, params.epsilon, params.Nsamples, params.Nart, episodes, etatheta, params.Algorithm == 3);
else
    [a, A, cov, rew, eta, theta, hist] = reps_state(a, sigs, contextFunc, simFunc, params.epsilon, params.Nsamples, episodes, etatheta);
end

% need to evaluate from history
linstat.R_mean = zeros(params.Niter, 1); 
linstat.evaluated = zeros(params.Niter, 1); 
linstat.theta_s = zeros(params.Niter, prod(params.Neval(1:s_dim))*th_dim);
linstat.R_s = zeros(params.Niter, prod(params.Neval(1:s_dim)));

for i=params.EvalModulo:params.EvalModulo:size(hist.a,1)
    %i=1+(params.EvalModulo-1)*params.Nsamples:params.EvalModulo*params.Nsamples:params.Niter
    a = hist.a(i,:)';
    A = reshape(hist.A(i,:)', th_dim, s_dim);
                
    theta_vec = zeros(prod(params.Neval(1:s_dim)),th_dim);
    val_vec = zeros(size(theta_vec,1),1);
    
    eval_s_vect = evalvectfun(params.xmin(1:s_dim), params.xmax(1:s_dim), params.Neval(1:s_dim));
    
    for j=1:size(eval_s_vect,1)
        theta = a + A*[eval_s_vect(j,:)'];
        
        theta_vec(j,:) = theta';
    end
    
    val_vec = problem.sim_eval_func([eval_s_vect theta_vec]);
 
    stat_i = 1+(i-1)*params.Nsamples;
    linstat.evaluated(stat_i,:) = 1;
    linstat.theta_s(stat_i, :) = theta_vec(:)';
    linstat.R_mean(stat_i, :) = mean(val_vec);
    linstat.R_s(stat_i,1) = rew(i);
    linstat.A(stat_i, :) = A(:);
    linstat.a(stat_i, :) = a(:);
    linstat.cov(stat_i, :) = hist.cov(i,:);
    %TODO use appropriate R naming, these are arbitrary
end


stats.last_R_mean = linstat.R_mean(end);


end


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
