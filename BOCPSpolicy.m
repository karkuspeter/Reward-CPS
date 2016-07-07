function [ theta ] = BOCPSpolicy( gprMdl, context, kappa, theta_bounds )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    f = @(x)(acq_func(gprMdl, [context, x'], kappa));
    [minval1,xatmin1,hist] = Direct(struct('f', f), theta_bounds);

    % refine by BFGS
    [xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton'));

    theta = xatmin2';
    theta = min(theta, theta_bounds(:,2));
    theta = max(theta, theta_bounds(:,1));

end

