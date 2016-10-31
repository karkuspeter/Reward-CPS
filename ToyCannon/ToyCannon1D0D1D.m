classdef ToyCannon1D0D1D < ToyCannonBaseES
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannon1D0D1D()
            obj = obj@ToyCannonBaseES();
 
            obj.theta_bounds = obj.theta_bounds(1,:);  %only angle
        end
        
        function r = r_func(obj, context, theta, outcome) % this does not effect return value of sim_func
           r = r_func@ToyCannonBaseES(obj, context, [theta 1], outcome); %rescale negative reward
        end
        
        function [r, result] = sim_func(obj, x) %includes full context, theta
            [r, result] = sim_func@ToyCannonBaseES(obj, [x(:,1), x(:,2), ones(size(x,1),1)]);
        end
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = sim_eval_func@ToyCannonBaseES(obj, [x(:,1), x(:,2), ones(size(x,1),1)]);
        end
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            theta_vec = zeros(contextpoints, size(obj.theta_bounds,1));
            r_vec = zeros(contextpoints, 1);
        end
    end
    
end

