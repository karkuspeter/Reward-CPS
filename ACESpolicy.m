function [ context, theta ] = ACESpolicy( gprMdl, params, context_bounds, theta_bounds )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

bounds = [context_bounds; theta_bounds];

f = @(x)(acq_func(gprMdl, x', params, kappa));


opts = cmaes('defaults');
opts.UBounds = bounds(:,2);
opts.LBounds = bounds(:,1);
opts.Restarts = 0;
opts.DispFinal = 'off';
opts.DispModulo = 'Inf';
%xstart = sprintf('%f + %f.*rand(%f,1)', theta_bounds(:,1), theta_bounds(:,2)-theta_bounds(:,1), size(theta_bounds,1)  );
xstart = mean(bounds, 2);
insigma = (bounds(:,2)-bounds(:,1))/3;

[xatmin1, minval1, counteval] = cmaes( ...
    f, ...    % name of objective/fitness function
    xstart, ...    % objective variables initial point, determines N
    insigma, ...   % initial coordinate wise standard deviation(s)
    opts ...    % options struct, see defopts below
    );

% refine by BFGS
[xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'none'));

theta = xatmin2';
theta = min(theta, theta_bounds(:,2));
theta = max(theta, theta_bounds(:,1));

end

