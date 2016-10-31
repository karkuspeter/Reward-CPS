classdef ToyCannon2D0D3D < ToyCannonBase3D
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyCannon2D0D3D()
            obj = obj@ToyCannonBase3D();
            
            obj.def_sigmaM0 = [0.6453    1.5913    1.2515]';
            obj.def_sigmaF0 = 1.4872;
        end
    end
    
end

