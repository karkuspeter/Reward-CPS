classdef ToyCannon1D2D < ProblemInterface
    %Toy cannon problem with only angle parameter to find
    properties
        toycannon;
    end
    
    methods
        function obj=ToyCannon1D2D()
            obj.toycannon = ToyCannon;

            obj.theta_bounds = [0, pi/2-0.2; 0.5, 1.5];
            obj.st_bounds = [0, 12];
            obj.se_bounds = [[], []];
        end
        
        function r = r_func(obj, context, theta, outcome)
           r = (obj.toycannon.r_func(theta(:,1), theta(:,2), context(:,1), outcome(:,1), outcome(:,2), outcome(:,3)) );
        end
        
        function [r, result] = sim_func(obj, context, theta)
            [r, result] = obj.toycannon.Simulate(context(:,1), theta(:,1), theta(:,2));
        end
        function [r, result] = sim_eval_func(obj, context, theta)
            [r, result] = obj.toycannon.Simulate(context(:,1), theta(:,1), theta(:,2), 0);
        end
        
        function [theta_vec, r_vec] = optimal_values(obj, datapoints)
            
            % compute optimal policy
            [x1, x2] = meshgrid(linspace(obj.st_bounds(1,1),obj.st_bounds(1,2), datapoints), ...
                linspace(obj.theta_bounds(1,1),obj.theta_bounds(1,2), datapoints));
            y = arrayfun(@(a,b)(obj.sim_eval_func(a,[b 1])), x1, x2)';
            [r_vec, theta_I] = max(y, [], 2);
            theta_vec = x2(theta_I);
        end
    end
    
end

