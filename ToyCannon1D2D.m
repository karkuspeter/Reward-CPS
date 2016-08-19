classdef ToyCannon1D2D < ProblemInterface
    %Toy cannon problem with only angle parameter to find
    properties
        toycannon;
        grid_cache; % cache sim results for evaluation
    end
    
    methods
        function obj=ToyCannon1D2D()
            obj.toycannon = ToyCannon;

            obj.theta_bounds = [0.01, pi/2-0.2; 0.1, 3];
            obj.st_bounds = [1, 11];
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
        
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            
            theta_vec = [];
            r_vec = [];
            [x1, x2] = ndgrid(linspace(obj.theta_bounds(1,1),obj.theta_bounds(1,2), datapoints), ...
                linspace(obj.theta_bounds(2,1),obj.theta_bounds(2,2), datapoints));
            
            for context = linspace(obj.st_bounds(1,1), obj.st_bounds(1,2), contextpoints)
                y = arrayfun(@(t1, t2)(obj.sim_eval_func(context, [t1 t2])), x1, x2);
                
                [r_opt, ind] = max(y(:)); %will return 1d indexing
                theta_opt = [x1(ind), x2(ind)];
                
                theta_vec = [theta_vec; theta_opt];
                r_vec = [r_vec; r_opt];
            end
            
        end
        
        function y = get_cached_grid(obj, context_id)
            

            if ~obj.grid_cache
                if exist('ToyCannon1D2Dcache.mat', 'file') == 2
                    obj.grid_cache = load('ToyCannon1D2Dcache.mat', 'grid_cache');
                end
                
                
            
            end
            y = obj.grid_cache(:,:,context_id);
        end
        
        function PlotEnv(obj)
            obj.toycannon.PlotEnv()
        end
    end
    
end

