classdef ProblemInterface
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        theta_bounds;
        st_bounds;
        se_bounds;
    end
    
    methods
        r = r_func(obj, context, theta, outcome);
        [r, result] = sim_func(obj, context, theta);
        [r, result] = sim_eval_func(obj, context, theta); % no noise version of sim_func
        [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints);
        PlotEnv(obj);
        obj = Randomise(obj, varargin);
    end
    
end

