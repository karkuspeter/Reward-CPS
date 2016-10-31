classdef ToyCannon1D0D8D < ToyCannon1D0D2D
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        extra_target;
        extra_r;
    end
    
    methods
        function obj=ToyCannon1D0D8D()
            obj = obj@ToyCannon1D0D2D();
 
            obj.theta_bounds = [obj.theta_bounds; [zeros(6,1), ones(6,1)]];
            
            obj.extra_target = [0.1 0.3 0.4 0.6 0.7 0.9];
            obj.extra_r = @(x)(sum(bsxfun(@minus,x,obj.extra_target).^2/length(obj.extra_target), 2));
        end
        
        function r = r_func(obj, context, theta, outcome) % this does not effect return value of sim_func
           r = r_func@ToyCannon1D0D2D(obj, context, theta(:,1:2), outcome); 
           r = r + obj.extra_r(theta(:,3:end));
        end
        
        function [r, result] = sim_func(obj, x) %includes full context, theta
            [r, result] = sim_func@ToyCannon1D0D2D(obj, x(:,1:3));
            r = r + obj.extra_r(x(:,4:end));
        end
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = sim_eval_func@ToyCannon1D0D2D(obj, x(:,1:3));
            r = r + obj.extra_r(x(:,4:end));
        end
    end
    
end

