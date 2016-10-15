classdef ToyBallSE < ToyBall
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj=ToyBallSE()
            obj = obj@ToyBall();
 
            % switch se and st bounds
            se = obj.se_bounds;
            obj.se_bounds = obj.st_bounds;
            obj.st_bounds = se;  
            
          obj.def_sigmaM0 = [0.6; 0.6; 0.2; 0.2];
            obj.def_sigmaF0 = 1;
            obj.def_sigma0 = 0.05;
        end        
    end
    
end

