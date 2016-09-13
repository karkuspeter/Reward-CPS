function [ theta ] = BOCPSpolicy( gprMdl, context, params, theta_bounds, use_cmaes )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    f = @(x)(acq_func_bo(gprMdl, [context, x'], params.kappa));
    
    if (use_cmaes)
        opts = cmaes('defaults');
        opts.UBounds = theta_bounds(:,2);
        opts.LBounds = theta_bounds(:,1);
        opts.Restarts = 0;
        opts.DispFinal = 'off';
        opts.DispModulo = 'Inf';
        %xstart = sprintf('%f + %f.*rand(%f,1)', theta_bounds(:,1), theta_bounds(:,2)-theta_bounds(:,1), size(theta_bounds,1)  );
        xstart = mean(theta_bounds, 2);
        insigma = (theta_bounds(:,2)-theta_bounds(:,1))/3;
        
        [xatmin1, minval1, counteval] = cmaes( ...
            f, ...    % name of objective/fitness function
            xstart, ...    % objective variables initial point, determines N
            insigma, ...   % initial coordinate wise standard deviation(s)
            opts ...    % options struct, see defopts below
        );
        
        %[minval0,xatmin0,hist] = Direct(struct('f', f), theta_bounds, struct('showits', 0));
        %if (minval1-minval0>0.02)
            %disp('DIRECT was better');
        %end
    else
        [minval1,xatmin1,hist] = Direct(struct('f', f), theta_bounds, struct('showits', 0));
    end
    
    % refine by BFGS
    [xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton', 'Display', 'none'));
    
    if any(xatmin2<theta_bounds(:,1)) || any(xatmin2>theta_bounds(:,2))
        %disp('warning: out of bounds');
        [xatmin2, minval2] = fmincon(f, xatmin1, [], [], [], [], theta_bounds(:,1), theta_bounds(:,2), [], optimoptions('fmincon', 'Display', 'none'));
        
    end
    if minval2 > minval1
        disp('warning: ACESpolicy minval1 < minval2');
    end
    
    theta = xatmin2;

    
%     theta = xatmin2;
%     theta = min(theta, theta_bounds(:,2));
%     theta = max(theta, theta_bounds(:,1));
    
    theta = theta';

end

