function [ theta, val_at_theta ] = ACESpolicy(f, theta_bounds)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

     [minval1,xatmin1,hist] = Direct(struct('f', f), theta_bounds, struct('showits', 0));
    
    % refine by BFGS
    [xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'none'));

    if any(xatmin2<theta_bounds(:,1)) || any(xatmin2>theta_bounds(:,2))
        disp('warning: out of bounds');
        [xatmin2, minval2] = fmincon(f, xatmin1, [], [], [], [], theta_bounds(:,1), theta_bounds(:,2), [], optimoptions('fmincon', 'Display', 'none'));

    end
    if minval2 > minval1
        disp('warning: ACESpolicy minval1 < minval2');
    end
    
    theta = xatmin2;
    %theta = min(theta, theta_bounds(:,2));
    %theta = max(theta, theta_bounds(:,1));
    % TODO this shoud give too much on the bounds. use fminconst instead
    
    theta = theta';
    val_at_theta = minval2;

end

