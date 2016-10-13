classdef ToyCannonBase3D < ToyCannonBase
    %Toy cannon problem with only angle parameter to find
    properties
    end
    
    methods
        function obj=ToyCannonBase3D()
            obj.toycannon = ToyCannonSimulator3D;

            obj.theta_bounds = [0.01, pi/2-0.2; ...
                                0, 2*pi;...
                                0.1, 3];
            obj.st_bounds = [-1.1, 1.1; -1.1, 1.1];
            obj.se_bounds = zeros(0,2);
        end
        
        function r = r_func(obj, context, theta, outcome)
            %a_vert, a_hor, v, s1, s2, hillats, xres1, xres2, yres
           r = (obj.toycannon.r_func(theta(:,1), theta(:,2), theta(:,3), context(:,1), context(:,2), outcome(:,1), outcome(:,2), outcome(:,3), outcome(:,4)) );
        end
        
        function [r, result] = sim_func(obj, x)
            [r, result] = obj.toycannon.Simulate(x(:,1)*10, x(:,2)*10, x(:,3), x(:,4), x(:,5));
        end
        
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = obj.toycannon.Simulate(x(:,1)*10, x(:,2)*10, x(:,3), x(:,4), x(:,5), 0);
        end
        
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            theta_vec = zeros(contextpoints, size(obj.theta_bounds,1));
            r_vec = zeros(contextpoints, 1);
        end
        
        function [r_opt, r_worse] = get_optimal_r(obj, Neval)
            r_opt = 0;
            r_worse = 2;
        end

    end
    
end

