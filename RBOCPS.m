% Acqusition function could use max(sigma) or sum(sigma) instead of local sigma.
% ITs just expensive to compute everywhere

% we could sample according to acqusition function, not just choose
% max? maybe doesnt make sense.. 

function [ r_mean ] = RBOCPS( input_params )
%Implements BO-CPS
%   Detailed explanation goes here

params = struct(...
     'kappa', 1, ...
     'sigma0', 0.2, ...
     'sigmaM0', 0.1, ...
     'Niter', 50, ...
     'output_off', 0);

theta_dim = 1;
s_dim = 1;
theta_bounds = [0, pi/2-0.2];
s_bounds = [0, 12];
bounds = [s_bounds; theta_bounds];

if (exist('input_params'))
    params = ProcessParams(params, input_params);
end

toycannon = ToyCannon;
sim_func = @(a,s)(toycannon.Simulate(s, a, 1));
sim_nonoise = @(a,s)(toycannon.Simulate(s, a, 1, 0));

%optional GP parameters
sigma0 = params.sigma0;
sigmaF0 = sigma0;
sigmaM0star = params.sigmaM0*[theta_bounds(:,2) - theta_bounds(:,1)];

% compute optimal policy
[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
    linspace(bounds(2,1),bounds(2,2), 100));
y = arrayfun(sim_nonoise, x2, x1)';
[r_opt, theta_I] = max(y, [], 2);
theta_opt = x2(theta_I);

Dfull = [];

for iter=1:params.Niter
    % sample random context
    context = ((s_bounds(:,2)-s_bounds(:,1)).*rand(size(s_bounds,1),1) + s_bounds(:,1))';
    
    % get prediction for context
    if(iter > 1)
        % map data
        Dstar = MapToContext(Dfull, context, toycannon.r_func);
        
        gprMdl = fitrgp(Dstar(:,1:end-1),Dstar(:,end),'Basis','constant','FitMethod','exact',...
            'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
            'KernelParameters',[sigmaM0star;sigmaF0], 'Sigma',sigma0,'Standardize',1);
        
        theta = BOCPSpolicy(gprMdl, [], params.kappa, theta_bounds);
        
    else
        theta = ((theta_bounds(:,2)-theta_bounds(:,1)).*rand(size(theta_bounds,1),1) + theta_bounds(:,1))';
    end
    
    % get sample for simulator
    [r, result] = sim_func(theta, context);
    
    % add to data matrix
    % D = [D; [context, theta, r]];
    Dfull = [Dfull; [theta, 1, context, result, r]];
    
    if (iter < 2)
        continue;
    end

    % evaluate offline performance
    context_vec = linspace(s_bounds(:,1), s_bounds(:,2), 100)';
    theta_space = linspace(theta_bounds(:,1),theta_bounds(:,2), 100)';
    ypred = []; ystd = []; acq_val=[];
    for i=1:size(context_vec,1)
        Dstar = MapToContext(Dfull, context_vec(i,:), toycannon.r_func);
        
        gprMdl = fitrgp(Dstar(:,1:end-1),Dstar(:,end),'Basis','constant','FitMethod','exact',...
            'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
            'KernelParameters',[sigmaM0star;sigmaF0], 'Sigma',sigma0,'Standardize',1);
        
        theta_vec(i,:) = BOCPSpolicy(gprMdl, [], 0, theta_bounds);
        r_vec(i,:) = sim_nonoise(theta_vec(i,:), context_vec(i,:));
        
        [newypred, newystd] = predict(gprMdl, theta_space);

        ypred = [ypred; newypred];
        ystd = [ystd; newystd];
        acq_val = [acq_val; -acq_func(gprMdl, theta_space, params.kappa)];
    end
    r_mean(iter) = mean(r_vec);
    
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

figure
plot(1:length(r_mean), r_mean, 1:length(r_mean), mean(r_opt)*ones(size(r_mean)))

end

function Dstar = MapToContext(Dfull, context, r_func)
Dstar = [Dfull(:,1), ...
    arrayfun(r_func, Dfull(:,1), Dfull(:,2), context*ones(size(Dfull,1),1), Dfull(:,4),Dfull(:,5),Dfull(:,6)) ];
end