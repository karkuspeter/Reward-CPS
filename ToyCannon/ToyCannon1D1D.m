classdef ToyCannon1D1D < ProblemInterface
    %Toy cannon problem with only angle parameter to find
    properties
        toycannon;
    end
    
    methods
        function obj=ToyCannon1D1D()
            obj.toycannon = ToyCannon;

            obj.theta_bounds = [0, pi/2-0.2];
            obj.st_bounds = [0, 12];
            obj.se_bounds = [[], []];
        end
        
        function r = r_func(obj, context, theta, outcome)
           r = (obj.toycannon.r_func(theta, 1, context, outcome(:,1), outcome(:,2), outcome(:,3)) );
        end
        
        function [r, result] = sim_func(obj, context, theta)
            [r, result] = obj.toycannon.Simulate(context, theta, 1);
        end
        function [r, result] = sim_eval_func(obj, context, theta)
            [r, result] = obj.toycannon.Simulate(context, theta, 1, 0);
        end
        
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            
            % compute optimal policy
            [x1, x2] = meshgrid(linspace(obj.st_bounds(1,1),obj.st_bounds(1,2), contextpoints), ...
                linspace(obj.theta_bounds(1,1),obj.theta_bounds(1,2), datapoints));
            y = arrayfun(@obj.sim_eval_func, x1, x2)';
            [r_vec, theta_I] = max(y, [], 2);
            theta_vec = x2(theta_I);
        end
        
        function PlotEnv(obj)
            obj.toycannon.PlotEnv()
        end
    end
    
end

