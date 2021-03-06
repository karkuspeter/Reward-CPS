classdef ToyCannonBase < ProblemInterface
    %Toy cannon problem with only angle parameter to find
    properties
        toycannon;
        grid_cache; % cache sim results for evaluation
        def_sigmaM0; % lengthscale (std, not cov)
        def_sigmaF0; % how much inputs are correlated - (std, not cov)
        def_sigma0; % noise level on signals (std, not cov);
    end
    
    methods
        function obj=ToyCannonBase()
            obj.toycannon = ToyCannonSimulator;

            obj.theta_bounds = [0.01, pi/2-0.2; 0.1, 3];
            obj.st_bounds = [1, 11];
            obj.se_bounds = zeros(0,2);

            obj.def_sigmaM0 = 0.4;
            obj.def_sigmaF0 = 1.5;
            obj.def_sigma0 = 0.0548;
        end
        
        function r = r_func(obj, context, theta, outcome)
           r = (obj.toycannon.r_func(theta(:,1), theta(:,2), context(:,1), outcome(:,1), outcome(:,2), outcome(:,3)) );
        end
        
        function [r, result] = sim_func(obj, context, theta)
            [r, result] = obj.toycannon.Simulate(context(:,1), theta(:,1), theta(:,2));
        end
        function [r, result] = sim_eval_func(obj, context, theta)
            [r, result] = obj.toycannon.Simulate(context(:,1), theta(:,1), theta(:,2), 0);
        end
              
        function [r, result] = sim_plot_func(obj, varargin)
            [r, result] = obj.sim_eval_func(varargin{:});
        end
      
        function [theta_vec, r_vec] = optimal_values(obj, datapoints, contextpoints)
            
            theta_vec = [];
            r_vec = [];
            [x1, x2] = ndgrid(linspace(obj.theta_bounds(1,1),obj.theta_bounds(1,2), datapoints), ...
                linspace(obj.theta_bounds(2,1),obj.theta_bounds(2,2), datapoints));           
             
            for context_id = 1:contextpoints
                y = obj.get_cached_grid(context_id, datapoints, contextpoints);
                [r_opt, ind] = max(y(:)); %will return 1d indexing
                theta_opt = [x1(ind), x2(ind)];
                
                theta_vec = [theta_vec; theta_opt];
                r_vec = [r_vec; r_opt];
            end
            
        end
        
        function y = get_cached_grid(obj, context_id, datapoints, contextpoints)
            

            if isempty(obj.grid_cache)
                if exist('ToyCannon1D2Dcache.mat', 'file') == 2
                    load('ToyCannon1D2Dcache.mat', 'grid_cache');
                    obj.grid_cache = grid_cache;
                else
                    obj.grid_cache = [];
                    [x1, x2] = ndgrid(linspace(obj.theta_bounds(1,1),obj.theta_bounds(1,2), datapoints), ...
                        linspace(obj.theta_bounds(2,1),obj.theta_bounds(2,2), datapoints));
                    
                    for context = linspace(obj.st_bounds(1,1), obj.st_bounds(1,2), contextpoints)
                        y = arrayfun(@(t1, t2)(obj.sim_eval_func(context, [t1 t2])), x1, x2);
                        obj.grid_cache(:,:,end+1) = y;
                    end
                    
                    grid_cache = obj.grid_cache;
                    save('ToyCannon1D2Dcache.mat', 'grid_cache');
                end
                
                
            end
            y = obj.grid_cache(:,:,context_id);
        end
        
        function PlotEnv(obj)
            obj.toycannon.PlotEnv()
        end
        
                
        function GPnew = MapGP(obj, GP, st, LearnHypers)
            if nargin < 4
                LearnHypers = false;
            end
            
            GPnew = GP;
            if size(st)
                for i=1:size(GP.x,1)
                    GPnew.y(i,:) = obj.r_func([st GP.x(i,1:size(obj.se_bounds,1))], GP.x(i,1+size(obj.se_bounds,1):end), GP.obs(i,:));
                end
            end
            
            if LearnHypers
                minimizeopts.length    = 50;
                minimizeopts.verbosity = 0;
                %GPnew.hyp = minimize(GP.hyp_initial,@(x)GP.HyperPrior(x,GPnew.x,GPnew.y),minimizeopts);
                
                GPnew.hyp = minimize(GP.hyp_initial,@(x)gp(x, GP.inf, [], GP.covfunc, GP.likfunc, GPnew.x, GPnew.y),minimizeopts);
                if isnan(GPnew.hyp.lik)
                    GPnew.hyp = GP.hyp_initial;
                    disp('Optimizing hyperparameters failed (not pos def matrix?)');
                end
                
                %fprintf 'hyperparameters optimized.'
                %display(['length scales: ', num2str(exp(GPnew.hyp.cov(1:end-1)'))]);
                %display([' signal stdev: ', num2str(exp(GPnew.hyp.cov(end)))]);
                %display([' noise stddev: ', num2str(exp(GPnew.hyp.lik))]);        
            end

            if ~isempty(st) || LearnHypers
                GPnew.K              = k_matrix(GPnew,GPnew.x) + diag(GP_noise_var(GPnew,GPnew.y));
                GPnew.cK             = robustchol(GPnew.K);
            end
        end

        
        function obj = Randomise(obj, varargin)
            %input: number of hills or goes to default
            obj.toycannon.Randomise(varargin{:});
        end
        
        function obj = SetRcoeff(obj, rcoeff)
            obj.toycannon.SetRcoeff(rcoeff);
        end
        
        function [r_opt, r_worse] = get_optimal_r(obj, Neval)
            evalgrid = evalgridfun(...
                [obj.st_bounds(:,1); obj.se_bounds(:,1); obj.theta_bounds(:,1)]', ...
                [obj.st_bounds(:,2); obj.se_bounds(:,2); obj.theta_bounds(:,2)]', ...
                Neval);
            val_full = arrayfun(@(varargin)(obj.sim_eval_func([varargin{:}])), evalgrid{:}); % this is a D dim array
            r_opt = val_full;
            r_worse = val_full;
            for i_th=size(obj.st_bounds,1)+size(obj.se_bounds,1)+1:ndims(val_full)
                r_opt = min(r_opt, [], i_th);
                r_worse = mean(r_worse, i_th);
            end
            
        end
        
    end
    
end

