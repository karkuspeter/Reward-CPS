classdef ToyBall < ProblemInterface
    % Class for toy cannon problem
    %   2D cannon with random hills
    
    %TODO dont return hills.. and no need for reward function
    
    properties
        scale
        noise
        cost
        PrintOn;
        step; %step size for simluation
        def_sigmaF0
        def_sigmaM0
        def_sigma0
    end
    
    methods
        function obj = ToyBall()
            obj.scale = [5 9; ... %st dist
                    ...-2.5 2.5
                    0 2; ... %st height
                    30, 60; ... %vertical angle
                    ...-45, 45;... %horizontal angle
                    2, 4]; % initial speed

            obj.theta_bounds = [0, 1; 0, 1];
            obj.st_bounds = [0, 1; 0, 1];
            obj.se_bounds = zeros(0,2);
            
            obj.step = 0.01;
            obj.noise = [0.01; 0.01];
            
            obj.cost = @(d2, th)((min(d2,1) + 0.05*sqrt(d2) + 0.004*180/pi*th(:,1)));

            obj.PrintOn = false;
    
            obj.def_sigmaM0 = [0.2; 0.2];
            obj.def_sigmaF0 = 1;
            obj.def_sigma0 = 0.05;
        end
        
        function PlotEnv(obj)
            [x1, x2] = ndgrid(linspace(obj.s_bounds(1,1)*2,obj.s_bounds(1,2)*2, 100), ...
                linspace(obj.s_bounds(2,1)*2,obj.s_bounds(2,2)*2, 100));
            y = arrayfun(@obj.HillValue, x1, x2);
            mesh(x1, x2, y);
        end
        
        function r = r_func(obj, context, theta, outcome)
            r = obj.Simulate([context, outcome(:,1:2)]);
        end
        
        function [r, result] = sim_func(obj, x)
            [r, result] = obj.Simulate(x, true);
        end
        
        function [r, result] = sim_eval_func(obj, x)
            [r, result] = obj.Simulate(x);
        end  
        function [r, result] = sim_plot_func(obj, x)
            [r, result] = obj.sim_eval_func(x);
        end  
        
        function [r, result] = Simulate(obj, xnorm, noiseOn)
            if nargin < 3
                noiseOn = false;
            end
            if noiseOn
                noise = obj.noise;
            else
                noise = 0;
            end
            
            if(obj.PrintOn)
                figure
                %obj.PlotEnv();
            end
            
            xnorm(:,3:4) = xnorm(:,3:4) + bsxfun(@times, randn(size(xnorm,1),2), noise');
           
            x=xnorm;
            x(:,:) = bsxfun(@plus, bsxfun(@times, x, (obj.scale(:,2)'-obj.scale(:,1)')), obj.scale(:,1)');

            sh = x(:,1);
            sv = x(:,2);
            th_v = x(:,3)*pi/180;
            v = x(:,4);
            cos_th_v = cos(th_v);
            sin_th_v = sin(th_v);
            
            t = 0:0.05:25;
            t = repmat(t, size(v));
            xhor = bsxfun(@times, v .* cos_th_v, t);
            xvert = bsxfun(@times, v .* sin_th_v, t) - 0.5 * 0.981 * t.^2;
            
            xhor(xvert < 0) = 0;
            %hiti = find(xhor<0, 1);

            d2all = bsxfun(@minus,sv,xvert).^2 + bsxfun(@minus,sh,xhor).^2;
            [dist2,ind] = min(d2all, [], 2);
            ind = sub2ind(size(xvert),1:length(ind), ind');
            if any(xvert(ind) < 0)
                disp('closest point is under surface');
            end
            
            r = obj.cost(dist2, [th_v, v]);
            result = [xnorm(:,3:4) sqrt(dist2)];
           
            if (obj.PrintOn)
                plot(xhor', xvert');
                hold on;
                scatter(sh, sv);
                scatter(xhor(ind), xvert(ind));
                ylim([0, 10]);
                xlim([0, 15]);
            end
            
        end
        function [r_opt, r_worse] = get_optimal_r(obj, Neval)
            stvec=linspace(0,1,Neval(1));
            
            r_opt = 0;
            r_worse = 100;
        end
        
        function obj = Randomise(obj)
        end
        function obj = SetRcoeff(obj)
        end
        
         function GPnew = MapGP(obj, GP, st, LearnHypers)
            if nargin < 4
                LearnHypers = false;
            end
            
            GPnew = GP;
            if size(st)
                GPnew.y(:,:) = obj.r_func([repmat(st,size(GP.x,1),1) GP.x(:,1:size(obj.se_bounds,1))], GP.x(:,1+size(obj.se_bounds,1):end), GP.obs(:,:));
            end
            
            if LearnHypers
                minimizeopts.length    = 50;
                minimizeopts.verbosity = 0;
                hp = @SEGammaHyperPosterior;
                GPnew.hyp = minimize(GP.hyp_initial,@(x)hp(x,GPnew.x,GPnew.y),minimizeopts);
                
                %GPnew.hyp = minimize(GP.hyp_initial,@(x)gp(x, GP.inf, [], GP.covfunc, GP.likfunc, GPnew.x, GPnew.y),minimizeopts);
                if isnan(GPnew.hyp.lik)
                    GPnew.hyp = GP.hyp_initial;
                    disp('Optimizing hyperparameters failed (not pos def matrix?)');
                end
            end

            if ~isempty(st) || LearnHypers
                GPnew.K              = k_matrix(GPnew,GPnew.x) + diag(GP_noise_var(GPnew,GPnew.y));
                GPnew.cK             = robustchol(GPnew.K);
            end
        end

    end
    
end