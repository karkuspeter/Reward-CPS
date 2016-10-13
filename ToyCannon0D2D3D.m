classdef ToyCannon0D2D3D < ToyCannon2D0D3D
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannon0D2D3D()
            obj = obj@ToyCannon2D0D3D();
 
            % switch se and st bounds
            se = obj.se_bounds;
            obj.se_bounds = obj.st_bounds;
            obj.st_bounds = se;            
        end        
    end
    
end

