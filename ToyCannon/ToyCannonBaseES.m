classdef ToyCannonBaseES < ToyCannonBase
    %Toy cannon problem with only angle parameter to find
    properties
    end
    
    methods
        function obj=ToyCannonBaseES()
            obj = obj@ToyCannonBase();
 
            obj.theta_bounds = obj.theta_bounds(:,:);
            obj.st_bounds = obj.st_bounds/10; %rescale
            obj.se_bounds = obj.se_bounds/10;
        end
        
        function r = r_func(obj, context, theta, outcome) % this does not effect return value of sim_func
           r = -r_func@ToyCannonBase(obj, context*10, theta, outcome)*0.2-0.4; %rescale negative reward
        end
        
        function [r, result] = sim_func(obj, x) %includes full context, theta
            [r, result] = sim_func@ToyCannonBase(obj, x(:,1)*10, [x(:,2) x(:,3)]);
            r = -r*0.2-0.4;
        end
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = sim_eval_func@ToyCannonBase(obj, x(:,1)*10, [x(:,2) x(:,3)]);
            r = -r*0.2-0.4;
        end
       
       end
    
end

