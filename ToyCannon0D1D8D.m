classdef ToyCannon0D1D8D < ToyCannon1D0D8D
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannon0D1D8D()
            obj = obj@ToyCannon1D0D8D();
 
            % switch se and st bounds
            se = obj.se_bounds;
            obj.se_bounds = obj.st_bounds;
            obj.st_bounds = se;            
        end        
    end
    
end

