% Acqusition function could use max(sigma) or sum(sigma) instead of local sigma.
% ITs just expensive to compute everywhere

% we could sample according to acqusition function, not just choose
% max? maybe doesnt make sense.. 

function [ stats, linstat, params ] = RBOCPS( input_params )
%Implements BO-CPS
%   Detailed explanation goes here

params = struct(...
     'kappa', 1, ...
     'sigma0', 0.2, ...
     'sigmaM0', 0.1, ...
     'Algorithm', 2, ...   % 1 BOCPS, 2 RBOCPS, 3 ACES, 4 RACES
     'Niter', 10, ...
     'EvalModulo', 1, ...
     'output_off', 0);
 
if (exist('input_params'))
    params = ProcessParams(params, input_params);
end

% isRBOCPS: whether running the proposed reward exploiting version
isRBOCPS = (params.Algorithm == 2 || params.Algorithm == 4);
isACES = (params.Algorithm == 3 || params.Algorithm == 4);

theta_dim = 1;
s_dim = 1;
theta_bounds = [0, pi/2-0.2];
s_bounds = [0, 12];
bounds = [s_bounds; theta_bounds];


toycannon = ToyCannon;
sim_func = @(a,s)(toycannon.Simulate(s, a, 1));
sim_nonoise = @(a,s)(toycannon.Simulate(s, a, 1, 0));

%optional GP parameters
sigma0 = params.sigma0;
sigmaF0 = sigma0;
sigmaM0 = params.sigmaM0 * [theta_bounds(:,2) - theta_bounds(:,1); ...
    s_bounds(:,2) - s_bounds(:,1)];
sigmaM0star = params.sigmaM0*[theta_bounds(:,2) - theta_bounds(:,1)];

% compute optimal policy
[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
    linspace(bounds(2,1),bounds(2,2), 100));
y = arrayfun(sim_nonoise, x2, x1)';
[r_opt, theta_I] = max(y, [], 2);
theta_opt = x2(theta_I);

Dfull = [];

stats = struct();
linstat = struct('R_mean', [], 's', [], 'theta', [], ...
                 'theta_s', [], 'R_s', [], 'R_opt', [], ...
                 'trial_result', []);

for iter=1:params.Niter
    % sample random context
    context = ((s_bounds(:,2)-s_bounds(:,1)).*rand(size(s_bounds,1),1) + s_bounds(:,1))';
    
    %override if reproducing previous results
    load('results/hyper-12-08-2016-22-19.mat', 'linstat_vec')
    context = linstat_vec(65).s(iter);
    
    % get prediction for context
    if(iter > 3)
        if isRBOCPS
            % map data
            Dstar = MapToContext(Dfull, context, toycannon.r_func);

            gprMdl = fitrgp(Dstar(:,1:end-1),Dstar(:,end),'Basis','constant','FitMethod','exact',...
                'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
                'KernelParameters',[sigmaM0star;sigmaF0], 'Sigma',sigma0,'Standardize',1);
            
            theta = BOCPSpolicy(gprMdl, [], params, theta_bounds, isACES);
        else
            % map data to have [context, theta, r]
            D = [Dfull(:,3), Dfull(:,1), Dfull(:,end)];
            
            % train GP
            gprMdl = fitrgp(D(:,1:end-1),D(:,end),'Basis','constant','FitMethod','exact',...
                'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
                'KernelParameters',[sigmaM0;sigmaF0], 'Sigma',sigma0,'Standardize',1);
            
            
            theta = BOCPSpolicy(gprMdl, context, params, theta_bounds, isACES);
        end
        
    else
        theta = ((theta_bounds(:,2)-theta_bounds(:,1)).*rand(size(theta_bounds,1),1) + theta_bounds(:,1))';
    end
    
    % get sample from simulator
    [r, result] = sim_func(theta, context);
    
    % add to data matrix
    Dfull = [Dfull; [theta, 1, context, result, r]];
    
    % add to stats
    linstat.s(iter,:) = context;
    linstat.theta(iter,:) = theta;
    linstat.r(iter,:) = r;
    linstat.R_opt(iter,:) = mean(r_opt); %this is always the same
    linstat.trial_result(iter, :) = result;
    
    if (iter < 4 || mod(iter, params.EvalModulo) > 0)
        continue;
    end
    
    % evaluate offline performance
    if (~params.output_off)
        fprintf('Eval iteration %d \n', iter);
    end
    context_vec = linspace(s_bounds(:,1), s_bounds(:,2), 100)';
    
    theta_space = linspace(theta_bounds(:,1),theta_bounds(:,2), 100)';
    ypred = []; ystd = []; acq_val=[];
    for i=1:size(context_vec,1)
        if isRBOCPS
            Dstar = MapToContext(Dfull, context_vec(i,:), toycannon.r_func);
            
            gprMdl = fitrgp(Dstar(:,1:end-1),Dstar(:,end),'Basis','constant','FitMethod','exact',...
                'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
                'KernelParameters',[sigmaM0star;sigmaF0], 'Sigma',sigma0,'Standardize',1);
            
            theta_vec(i,:) = BOCPSpolicy(gprMdl, [], struct('kappa',0) , theta_bounds, false);
            pred_space = theta_space;
        else           
            theta_vec(i,:) = BOCPSpolicy(gprMdl, context_vec(i,:), struct('kappa',0), theta_bounds, false);  
            pred_space = [context_vec(i,:)*ones(size(theta_space)), theta_space];
        end
        r_vec(i,:) = sim_nonoise(theta_vec(i,:), context_vec(i,:));
        [newypred, newystd] = predict(gprMdl, pred_space);
        
        ypred = [ypred; newypred];
        ystd = [ystd; newystd];
        acq_val = [acq_val; -acq_func_bo(gprMdl, pred_space, params.kappa)];
    end
    
    linstat.theta_s(iter,:) = theta_vec';
    linstat.R_s(iter,:) = r_vec';
    linstat.R_mean(iter,:) = mean(r_vec);
    
    if params.output_off
        continue;
    end
    
    % show environment and performance
    [x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
        linspace(bounds(2,1),bounds(2,2), 100));
    y = arrayfun(sim_nonoise, x2, x1);
    figure(1);
    mesh(x1(1,:)', x2(:,1), y);
    hold on
    plot3(context_vec, theta_vec, r_vec);
    plot3(x1(1,:)', theta_opt, r_opt);
    hold off
    xlabel('context');
    ylabel('angle');
    view(0,90)
    %legend('Real R values');
    
    % show prediction
    %[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
    %    linspace(bounds(2,1),bounds(2,2), 100));
    %Xplot = reshape(cat(2, x1, x2), [], 2);
    %[ypred, ystd] = predict(gprMdl, Xplot);
    Yplot = reshape(ypred, size(x1,1), []);
    
    figure(2);
    mesh(x1(1,:)', x2(:,1), Yplot);
    hold on
    scatter3(Dfull(:,3), Dfull(:,1), Dfull(:,7));
    xlabel('context');
    ylabel('angle');
    %legend('Data','GPR predictions');
    view(0,90)
    hold off
    
    figure(3);
    Yplot = reshape(ystd, size(x1,1), []);
    mesh(x1(1,:)', x2(:,1), Yplot);
    xlabel('context');
    ylabel('angle');
    %legend('GPR uncertainty');
    view(0,90)
    hold off
    
    Yplot = reshape(acq_val, size(x1,1), []);
    
    figure(4);
    mesh(x1(1,:)', x2(:,1), Yplot);
    hold on
    scatter3(Dfull(:,3), Dfull(:,1), Dfull(:,7));
    xlabel('context');
    ylabel('angle');
    %legend('Data','Aquisition function');
    view(0,90)
    hold off
    
    drawnow
    %pause;
end

if ~params.output_off
    figure
    plot(1:length(linstat.R_mean), linstat.R_mean, ...
         1:length(linstat.R_mean), mean(r_opt)*ones(size(linstat.R_mean)))
end

end

function Dstar = MapToContext(Dfull, context, r_func)
Dstar = [Dfull(:,1), ...
    arrayfun(r_func, Dfull(:,1), Dfull(:,2), context*ones(size(Dfull,1),1), Dfull(:,4),Dfull(:,5),Dfull(:,6)) ];
end