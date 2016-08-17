params = struct(...
     'problem', ToyCannon1D2D, ...
     'kappa', 1, ...
     'sigma0', 0.2, ...
     'sigmaM0', 1, ...
     'Algorithm', 1, ...   % 1 BOCPS, 2 RBOCPS, 3 ACES, 4 RACES
     'Niter', 50, ...
     'EvalModulo', 1, ...
     'EvalAllTheta', 0, ...
     'output_off', 0);

 problem = params.problem;

% isRBOCPS: whether running the proposed reward exploiting version
isRBOCPS = (params.Algorithm == 2 || params.Algorithm == 4);
isACES = (params.Algorithm == 3 || params.Algorithm == 4);

theta_bounds = problem.theta_bounds;
st_bounds = problem.st_bounds;
se_bounds = problem.se_bounds;
bounds = [st_bounds; se_bounds; theta_bounds];
sfull_bounds = [st_bounds; se_bounds];
theta_dim = size(theta_bounds,1);
st_dim = size(st_bounds,1);
se_dim = size(se_bounds,1);
%TODO simple BOCPS should be to move st bounds to se bounds and thats all

%optional GP parameters
Dfull = struct('st', [], 'se', [], 'theta', [], 'outcome', [], 'r', []);

stats = struct('last_R_mean', 0);
linstat = struct('R_mean', [], 'st', [], 'se', [], 'theta', [], ...
                 'theta_s', [], 'R_s', [], 'R_opt', [], ...
                 'outcome', []);
context_vec = linspace(sfull_bounds(:,1), sfull_bounds(:,2), 100)';
        context_id = 50;
        context = context_vec(context_id);
        
rng(seed);
%% iter
for iter=1:5
        context_t = ((st_bounds(:,2)-st_bounds(:,1)).*rand(size(st_bounds,1),1) + st_bounds(:,1))';
        theta = ((theta_bounds(:,2)-theta_bounds(:,1)).*rand(size(theta_bounds,1),1) + theta_bounds(:,1))';
        
        context_t = context;
        
        [r, outcome] = problem.sim_eval_func([context_t], theta);
   % add to data matrix
    Dfull.st = [Dfull.st; context_t];
    %Dfull.se = [Dfull.se; context_e];
    Dfull.theta = [Dfull.theta; theta];
    Dfull.outcome = [Dfull.outcome; outcome];
    Dfull.r = [Dfull.r; r];
    
    % add to stats
    linstat.st = [linstat.st; context_t];
    %linstat.se = [linstat.se; context_e];
    linstat.theta(iter,:) = theta;
    linstat.r(iter,:) = r;
    linstat.R_opt(iter,:) = mean(r_opt); %this is always the same
    linstat.outcome(iter, :) = outcome; %rename this
    
end


%% train GP
sigmaM0 = [1; 1]; % how much inputs should be similar in that dim
sigmaF0 = 1;  % how much inputs are correlated - 
sigma0 = 0.1; %how noisy my observations are
gprMdl = fitrgp([Dfull.theta(:,1)/10000,Dfull.theta(:,2) ],Dfull.r, ...
    'Basis','constant','FitMethod','exact',...
    'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
    'KernelParameters',[sigmaM0;sigmaF0], ... %[length scales + max covariance between inputs]
    'Sigma',sigma0, ... %how noisy my observations are
    'Standardize',1); %if its standardized variance values are interpreted on standardized values
L = resubLoss(gprMdl)



        
[t1, t2] = ndgrid(linspace(theta_bounds(1,1),theta_bounds(1,2), 100), ...
    linspace(theta_bounds(2,1),theta_bounds(2,2), 100));  %means indexing output by the first dimension indexes into different theta(1) values
theta_space = [t1(:)/10000, t2(:)];

ypred = []; ystd = []; acq_val=[];
%pred_space = [context_vec(i,:)*ones(size(theta_space,1),size(context_vec,2)), theta_space];
pred_space = [theta_space];
[ypred, ystd] = predict(gprMdl, pred_space);
%acq_val = [acq_val; -acq_func_bo(gprMdl, pred_space, params.kappa)];
    
% for i=1:size(context_vec,1)
%     theta_vec(i,:) = BOCPSpolicy(gprMdl, context_vec(i,:), struct('kappa',0), theta_bounds, false);
%     pred_space = [context_vec(i,:)*ones(size(theta_space,1),size(context_vec,2)), theta_space];
%     r_vec(i,:) = problem.sim_eval_func(context_vec(i,:), theta_vec(i,:));
%     [newypred, newystd] = predict(gprMdl, pred_space);
%     
%     ypred = [ypred; newypred];
%     ystd = [ystd; newystd];
%     acq_val = [acq_val; -acq_func_bo(gprMdl, pred_space, params.kappa)];
%     
% end

        % x1_vec is a single value this case (single context)

        [x1, x2] = ndgrid(linspace(theta_bounds(1,1),theta_bounds(1,2), 100), ...
            linspace(theta_bounds(2,1),theta_bounds(2,2), 100));
        y = arrayfun(@(t1, t2)(problem.sim_eval_func(context, [t1 t2])), x1, x2);
          
        [ropt, ind] = max(y);
        %[ind] = ind2sub([100,100], ind);
        x1opt_vec = x1(ind);
        x2opt_vec = x2(ind);
        zopt_vec = ropt;
        
%Yplot = reshape(ypred, size(context_vec,1), 100, []);
Yplot = ypred;
%Yplot = Yplot(context_id, :, :);
Yplot = reshape(Yplot, 100, 100);
        
    figure(1);
    mesh(x1, x2, y);
    hold on
    scatter3(Dfull.theta(:,1), Dfull.theta(:,2), Dfull.r);
    hold off
    xlabel('context');
    ylabel('angle');
    colorscale = caxis();
    %view(0,90)

    
    figure(2);
    mesh(x1, x2, Yplot);
    hold on
    if theta_dim == 1
        scatter3(Dfull.st, Dfull.theta, Dfull.r);
    else
        scatter3(Dfull.theta(:,1), Dfull.theta(:,2), Dfull.r);
    end
    xlabel('context');
    ylabel('angle');
    caxis(colorscale);
     hold off
    
    figure(3);
    mesh(x1, x2, Yplot - y);
    hold on
    xlabel('context');
    ylabel('angle');
    hold off    

% 
% function Rstar = MapToContext(Dfull, context, r_func)
% Rstar = zeros(size(Dfull.theta,1),1);
% for i=1:size(Dfull.theta,1)
%     if size(Dfull.se)
%         sfull = [context, Dfull.se(i,:)];
%     else
%         sfull = context;
%     end
%     Rstar(i) = r_func(sfull, Dfull.theta(i,:), Dfull.outcome(i,:));
% end
% end