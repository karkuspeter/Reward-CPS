classdef ToyCannonS0D2D3D < ToyCannonS2D0D3D
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannonS0D2D3D()
            obj = obj@ToyCannonS2D0D3D();
 
            % switch se and st bounds
            se = obj.se_bounds;
            obj.se_bounds = obj.st_bounds;
            obj.st_bounds = se;  
            
            obj.def_sigmaM0 = [0.9469 0.9375 1.1843  1.7052 1.7179]';
            obj.def_sigmaF0 = 1.9085;
        end        
    end
    
end

