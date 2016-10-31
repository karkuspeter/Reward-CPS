classdef ToyCannonBase3D < ToyCannonBase
    %Toy cannon problem with only angle parameter to find
    properties
    end
    
    methods
        function obj=ToyCannonBase3D()
            obj.toycannon = ToyCannonSimulator3D;

            obj.theta_bounds = [0.01, pi/2-0.2; ...
                                0, 2*pi;...
                                0.1, 5];
            obj.st_bounds = [-1.1, 1.1; -1.1, 1.1];
            obj.se_bounds = zeros(0,2);
        end
        
        function r = r_func(obj, context, theta, outcome)
            %a_vert, a_hor, v, s1, s2, hillats, xres1, xres2, yres
           r = (obj.toycannon.r_func(theta(:,1), theta(:,2), theta(:,3), context(:,1)*10, context(:,2)*10, outcome(:,1), outcome(:,2), outcome(:,3), outcome(:,4)) );
        end
        
        function [r, result] = sim_func(obj, x)
            [r, result] = obj.toycannon.Simulate(x(:,1)*10, x(:,2)*10, x(:,3), x(:,4), x(:,5));
        end
        
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = obj.toycannon.Simulate(x(:,1)*10, x(:,2)*10, x(:,3), x(:,4), x(:,5), 0);
        end
        
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            theta_vec = zeros(contextpoints, size(obj.theta_bounds,1));
            r_vec = zeros(contextpoints, 1);
        end
        
        function [r_opt, r_worse] = get_optimal_r(obj, Neval)
            dim = size(obj.st_bounds,1)+size(obj.se_bounds,1);
            evalgrid = evalgridfun(...
                [obj.st_bounds(:,1); obj.se_bounds(:,1)]', ...
                [obj.st_bounds(:,2); obj.se_bounds(:,2)]', ...
                Neval(1:dim));
            r_opt = obj.get_optimal_for_s(evalgrid{:}, Neval(dim+1:end));
            r_worse = 2;
        end
        function [r_opt, r_worse] = get_optimal_for_s(obj, s1, s2, Neval)
            ang_h = atan2(s2, s1); 
            ang_h(ang_h<0) = ang_h(ang_h<0) + 2*pi;
            evalgrid = cell(1, 3);
            gridvect = cell(1, 3);
            gridvect{1} = linspace(obj.theta_bounds(1,1), obj.theta_bounds(1,2), Neval(1))';
            gridvect{2} = ang_h(:);
            gridvect{3} = linspace(obj.theta_bounds(3,1), obj.theta_bounds(3,2), Neval(3))';
            [evalgrid{:}] = ndgrid(gridvect{:});

            val_full = zeros(size(evalgrid{1}));
            for i = 1:size(evalgrid{1},1)
                for j = 1:size(evalgrid{1},2)
                    for k = 1:size(evalgrid{1},3)
                        val_full(i,j,k) = obj.sim_eval_func([s1(j), s2(j), evalgrid{1}(i,j,k), evalgrid{2}(i,j,k), evalgrid{3}(i,j,k)]);
                    end
                end
            end
            r_opt = val_full;
            r_opt = min(r_opt, [], 1);
            r_opt = min(r_opt, [], 3);
            r_opt = reshape(r_opt, size(ang_h));
  
%             f=@(x)obj.sim_eval_func([s, x']);
%             [minval1,xatmin1,hist] = Direct(struct('f', f), obj.theta_bounds, struct('showits', 1, 'maxevals', 200));
%             [xatmin2, minval2] = fmincon(f, xatmin1, [], [], [], [], obj.theta_bounds(:,1), obj.theta_bounds(:,2), [], optimoptions('fmincon', 'Display', 'none'));
            
            r_worse = 2;            
        end
        
    end
    
end

