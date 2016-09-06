classdef ToyCannon0D1D1D < ToyCannon1D0D1D
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannon0D1D1D()
            obj = obj@ToyCannon1D0D1D();
 
            % switch se and st bounds
            se = obj.se_bounds;
            obj.se_bounds = obj.st_bounds;
            obj.st_bounds = se;            
        end        
    end
    
end

