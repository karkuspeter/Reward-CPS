classdef ToyCannon1D1Des < ToyCannon1D2D
    %Toy cannon problem with only angle parameter to find
    properties
    end
    
    methods
        function obj=ToyCannon1D1Des()
            obj = obj@ToyCannon1D2D();
 
            obj.theta_bounds = obj.theta_bounds(1,:); %only angle
            obj.st_bounds = obj.st_bounds/10; %rescale
            obj.se_bounds = obj.se_bounds/10;
        end
        
        function r = r_func(obj, context, theta, outcome)
           r = -r_func@ToyCannon1D2D(obj, context*10, [theta 1], outcome)/5-0.4; %rescale negative reward
        end
        
        function [r, result] = sim_func(obj, x) %includes full context, theta
            [r, result] = sim_func@ToyCannon1D2D(obj, x(:,1)*10, [x(:,2) 1]);
        end
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = sim_eval_func@ToyCannon1D2D(obj, x(:,1)*10, [x(:,2) 1]);
        end

       end
    
end

