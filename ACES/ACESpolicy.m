function [ theta, val_at_theta ] = ACESpolicy( GP, context, theta_bounds)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

     f = @(theta)(gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, [context theta']));
     [minval1,xatmin1,hist] = Direct(struct('f', f), theta_bounds, struct('showits', 0));
    
    % refine by BFGS
    [xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'none'));

    theta = xatmin2;
    theta = min(theta, theta_bounds(:,2));
    theta = max(theta, theta_bounds(:,1));
    % TODO this shoud give too much on the bounds. use fminconst instead
    
    theta = theta';
    val_at_theta = minval2;

end

