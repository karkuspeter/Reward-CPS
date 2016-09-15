% Acqusition function could use max(sigma) or sum(sigma) instead of local sigma.
% ITs just expensive to compute everywhere

% we could sample according to acqusition function, not just choose
% max? maybe doesnt make sense.. 

function [ stats, linstat, params ] = RBOCPS( input_params )
%Implements BO-CPS
%   Detailed explanation goes here

params = struct(...
     'problem', ToyCannon1D0D2D, ...
     'kappa', 1.25, ...
     'sigmaM0', 0.45^2, ...%[0.01; 0.01],... %; 0.1], ... % lengthscale, how much inputs should be similar in that dim. 
     ...               % i.e. how far inputs should influence each other
     ...               % can be single value or vector for each theta dim
     'sigmaF0', 0.8,...  % how much inputs are correlated - 
     'sigma0', sqrt(0.003), ... %how noisy my observations are, ...
     'Algorithm', 2, ...   % 1 BOCPS, 2 RBOCPS, 3, PRBOCPS, 4 ACES, 5 RACES
     'Niter', 50, ...
     'InitialSamples', 4, ...
     'Neval', [100, 100], ... %evaluation points over contexts
     'EvalModulo', 5, ...
     'EvalAllTheta', 0, ...
     'PopulateN', 6, ...
     'output_off', 0);
 
if (exist('input_params'))
    params = ProcessParams(params, input_params);
end

%params.Niter = params.Niter + params.InitialSamples; % to make it consistent with entropy search

problem = params.problem;

% isRBOCPS: whether running the proposed reward exploiting version
isRBOCPS = (params.Algorithm == 2 || params.Algorithm == 3 || params.Algorithm == 5);
isPopulate = (params.Algorithm == 3);
isACES = (params.Algorithm == 4 || params.Algorithm == 5);

theta_bounds = problem.theta_bounds;
st_bounds = problem.st_bounds;
se_bounds = problem.se_bounds;
bounds = [st_bounds; se_bounds; theta_bounds];
sfull_bounds = [st_bounds; se_bounds];
theta_dim = size(theta_bounds,1);
st_dim = size(st_bounds,1);
se_dim = size(se_bounds,1);
%TODO simple BOCPS should be to move st bounds to se bounds and thats all
             
% convert length scales
if isRBOCPS && ~isPopulate
    gp_dim = se_dim+theta_dim;
    context_mask = (st_dim+1):(st_dim+se_dim);
else
    gp_dim = st_dim+se_dim+theta_dim;
    context_mask = 1:(st_dim+se_dim);
end
if isPopulate
    Npopulate = params.PopulateN;
else
    Npopulate = 0;
end
if length(params.sigmaM0) == 1
    
    params.sigmaM0 = params.sigmaM0 * ones(gp_dim, 1);
elseif length(params.sigmaM0) ~= gp_dim
    disp('sigmaM0 vector length mismatch');
    return;
end

% optimal values
[theta_opt, r_opt] = problem.optimal_values(100, 100);



%% main iter
    
Dfull = struct('st', [], 'se', [], 'theta', [], 'outcome', [], 'r', []);

stats = struct('last_R_mean', 0);
linstat = struct('R_mean', [], 'st', [], 'se', [], 'theta', [], ...
                 'theta_s', [], 'R_s', [], 'R_opt', [], ...
                 'outcome', [], 'evaluated', []);
             
for iter=1:params.Niter
    % sample random context
    if st_dim
        context_t = ((st_bounds(:,2)-st_bounds(:,1)).*rand(size(st_bounds,1),1) + st_bounds(:,1))';
    else
        context_t = [];
    end
    if se_dim
        context_e = ((se_bounds(:,2)-se_bounds(:,1)).*rand(size(se_bounds,1),1) + se_bounds(:,1))';
    else
        context_e = [];
    end
    context_full = [context_t, context_e];
    %override if reproducing previous results
    % this is unnccessary, random seed will produce the same anyway
    %load('results/hyper-12-08-2016-22-19.mat', 'linstat_vec')
    %context_t = linstat_vec(65).s(iter);
    
    % get prediction for context
    if(iter > params.InitialSamples)
        if isRBOCPS
            % map data
            [D, Rstar] = MapToContext(Dfull, context_t, @problem.r_func, isPopulate, Npopulate, st_bounds);
        else
            D = [Dfull.st, Dfull.se, Dfull.theta];
            Rstar = Dfull.r;
        end
        gprMdl = fitrgp(D, Rstar, ...
            'Basis','constant','FitMethod','exact',...
            'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
            'KernelParameters',[params.sigmaM0; params.sigmaF0],...
            'Sigma',params.sigma0, ...
            'SigmaLowerBound', 1e-1*std(Rstar), ...            
            'Standardize',0);
        theta = BOCPSpolicy(gprMdl, context_full(1,context_mask), params, theta_bounds, isACES);
        
    else
        theta = ((theta_bounds(:,2)-theta_bounds(:,1)).*rand(size(theta_bounds,1),1) + theta_bounds(:,1))';
    end
    
    % get sample from simulator
    [r, outcome] = problem.sim_func([context_t, context_e, theta]);
    
    % add to data matrix
    %if isPopulate
    %    new_data_count = 100;
    %    st_vec = linspace(st_bounds(1,1),st_bounds(1,2), new_data_count)';
    %    r_vec = arrayfun(@(st)(problem.r_func(st, theta, outcome )), st_vec);
    %else
        new_data_count = 1;
        st_vec = context_t;
        r_vec = r;
    %end
    
    Dfull.st = [Dfull.st; st_vec];
    Dfull.se = [Dfull.se; repmat(context_e, new_data_count, 1)];
    Dfull.theta = [Dfull.theta; repmat(theta, new_data_count, 1)];
    Dfull.outcome = [Dfull.outcome; repmat(outcome, new_data_count, 1)];
    Dfull.r = [Dfull.r; r_vec];
    
    % add to stats
    linstat.st = [linstat.st; context_t];
    linstat.se = [linstat.se; context_e];
    linstat.theta(iter,:) = theta;
    linstat.r(iter,:) = r;
    linstat.R_opt(iter,:) = mean(r_opt); %this is always the same
    linstat.outcome(iter, :) = outcome; %rename this
    linstat.evaluated(iter, :) = 0;
    
    if (mod(iter, params.EvalModulo) > 0 || iter<=params.InitialSamples)
        continue;
    end
    
    %% evaluate offline performance
    if (~params.output_off)
        fprintf('Eval iteration %d \n', iter);
    end
    context_vec = linspace(sfull_bounds(:,1), sfull_bounds(:,2), params.Neval(1))';
    
    if theta_dim == 1
        theta_space = linspace(theta_bounds(:,1),theta_bounds(:,2), 100)';
    elseif theta_dim == 2
        [t1, t2] = ndgrid(linspace(theta_bounds(1,1),theta_bounds(1,2), 100)', ...
            linspace(theta_bounds(2,1),theta_bounds(2,2), 100)');
        theta_space = [t1(:), t2(:)];
    else
        theta_space = [];
    end
    ypred = []; ystd = []; acq_val=[]; policy_pred_vec=[];
    for i=1:size(context_vec,1)
        if isRBOCPS
            [D, Rstar] = MapToContext(Dfull, context_vec(i,1:st_dim), @problem.r_func, isPopulate, Npopulate, st_bounds);
            
            gprMdl = fitrgp(D, Rstar, ...
                'Basis','constant','FitMethod','exact',...
                'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
                'KernelParameters',[params.sigmaM0; params.sigmaF0],...
                'Sigma',params.sigma0, ...
                'SigmaLowerBound', 1e-1*std(Rstar), ...
                'Standardize',1);
        end
            
        theta_vec(i,:) = BOCPSpolicy(gprMdl, context_vec(i,context_mask), struct('kappa',0) , theta_bounds, false);
        policy_pred_vec = [policy_pred_vec; predict(gprMdl, [context_vec(i,context_mask), theta_vec(i,:)])];
        if ~isempty(context_mask)
            pred_space = [repmat(context_vec(i,context_mask), size(theta_space,1),1), theta_space];
        else
            pred_space = theta_space;
        end
        r_vec(i,:) = problem.sim_eval_func([context_vec(i,:), theta_vec(i,:)]);
        
        %if params.EvalAllTheta
        [newypred, newystd] = predict(gprMdl, pred_space);
        
        ypred = [ypred; newypred];
        ystd = [ystd; newystd];
        acq_val = [acq_val; -acq_func_bo(gprMdl, pred_space, params.kappa)];
        %end
    end
    
    linstat.theta_s(iter,:) = theta_vec(:,1)';
    linstat.R_s(iter,:) = r_vec';
    linstat.R_mean(iter,:) = mean(r_vec);
    linstat.evaluated(iter,:) = 1; %override previous 0

    %% show environment and performance
    if (params.output_off || st_dim + se_dim + theta_dim > 3 || iter < 1)
        continue;
    end
    context_id = 20;
    ShowRBOCPS;

end

stats.last_R_mean = linstat.R_mean(end,:);

if ~params.output_off
    figure
    mask = 1:length(linstat.R_mean);
    mask = mask(linstat.evaluated > 0);
    plot(mask', linstat.R_mean(mask), ...
         mask', mean(r_opt).*ones(size(mask')))
     % params.EvalModulo:params.EvalModulo:length(linstat.R_mean)
end

end

