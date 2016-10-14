classdef ToyCannonS2D0D3D < ToyCannon2D0D3D
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannonS2D0D3D()
            obj = obj@ToyCannon2D0D3D();
            
            obj.theta_bounds = [0.01, pi/2-0.2; ...
                                0, pi/2;...
                                0.1, 3];
            obj.st_bounds = [0.1, 1.1; 0.1, 1.1];
            obj.se_bounds = zeros(0,2);
        end
    end
    
end

