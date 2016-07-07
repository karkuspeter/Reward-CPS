function [ r_mean ] = BOCPS( input_args )
%Implements BO-CPS
%   Detailed explanation goes here
kappa = 1;
theta_dim = 1;
s_dim = 1;
theta_bounds = [0, pi/2-0.2];
s_bounds = [0, 12];
bounds = [s_bounds; theta_bounds];
    
toycannon = ToyCannon;
sim_func = @(a,s)(toycannon.Simulate(s, a, 1));
sim_nonoise = @(a,s)(toycannon.Simulate(s, a, 1, 0));

%optional GP parameters
sigma0 = 0.2;
sigmaF0 = sigma0;
sigmaM0 = 0.1*[theta_bounds(:,2) - theta_bounds(:,1); ...
    s_bounds(:,2) - s_bounds(:,1)];

% compute optimal policy
[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
                        linspace(bounds(2,1),bounds(2,2), 100));
y = arrayfun(sim_nonoise, x2, x1)';
[r_opt, theta_I] = max(y, [], 2);
theta_opt = x2(theta_I);


D = [];


for iter=1:50
    % sample random context
    context = ((s_bounds(:,2)-s_bounds(:,1)).*rand(size(s_bounds,1),1) + s_bounds(:,1))';
    
    % get prediction for context
    if(iter > 1)
        theta = BOCPSpolicy(gprMdl, context, kappa, theta_bounds);
    else
        theta = ((theta_bounds(:,2)-theta_bounds(:,1)).*rand(size(theta_bounds,1),1) + theta_bounds(:,1))';
    end
    
    % get sample for simulator
    r = sim_func(theta, context);
    
    % add to data matrix
    D = [D; [context, theta, r]];
    
    % train GP
    gprMdl = fitrgp(D(:,1:end-1),D(:,end),'Basis','constant','FitMethod','exact',...
        'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
        'KernelParameters',[sigmaM0;sigmaF0], 'Sigma',sigma0,'Standardize',1);
    
    % evaluate offline performance
    context_vec = linspace(s_bounds(:,1), s_bounds(:,2), 100)';
    for i=1:size(context_vec,1)
        theta_vec(i,:) = BOCPSpolicy(gprMdl, context_vec(i,:), 0, theta_bounds);
        r_vec(i,:) = sim_nonoise(theta_vec(i,:), context_vec(i,:));
    end
    r_mean(iter) = mean(r_vec);
    
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
    [x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
        linspace(bounds(2,1),bounds(2,2), 100));
    Xplot = reshape(cat(2, x1, x2), [], 2);
    [ypred, ystd] = predict(gprMdl, Xplot);
    Yplot = reshape(ypred, size(x1,1), []);
    
    figure(2);
    mesh(x1(1,:)', x2(:,1), Yplot);
    hold on
    scatter3(D(:,1), D(:,2), D(:,3));
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
    
    acq_val = -acq_func(gprMdl, Xplot, kappa);
    Yplot = reshape(acq_val, size(x1,1), []);
    
    figure(4);
    mesh(x1(1,:)', x2(:,1), Yplot);
    hold on
    scatter3(D(:,1), D(:,2), D(:,3));
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
