classdef ToyCannon1D0D2D < ToyCannonBaseES
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannon1D0D2D()
            obj = obj@ToyCannonBaseES();
        end
        
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            theta_vec = zeros(contextpoints, size(obj.theta_bounds,1));
            r_vec = zeros(contextpoints, 1);
        end
    end
    
end

