function out = MyEntropySearch(in)
% probabilistic line search algorithm that adapts it search space
% stochastically, by sampling search points from their marginal probability of
% being smaller than the current best function guess.
%
% (C) Philipp Hennig & Christian Schuler, August 2011

fprintf 'starting entropy search.\n'

%% fill in default values where possible
if ~isfield(in,'likfunc'); in.likfunc = @likGauss; end; % noise type
if ~isfield(in,'poly'); in.poly = -1; end; % polynomial mean? 
if ~isfield(in,'log'); in.log = 0; end;  % logarithmic transformed observations?
if ~isfield(in,'with_deriv'); in.with_deriv = 0; end; % derivative observations?
if ~isfield(in,'x'); in.x = []; end;  % prior observation locations
if ~isfield(in,'y'); in.y = []; end;  % prior observation values
if ~isfield(in,'T'); in.T = 200; end; % number of samples in entropy prediction
if ~isfield(in,'Ne'); in.Ne = 4; end; % number of restart points for search
if ~isfield(in,'Nb'); in.Nb = 50; end; % number of representers
if ~isfield(in,'LossFunc'); in.LossFunc = {@LogLoss}; end;
if ~isfield(in,'PropFunc'); in.PropFunc = {@EI_fun}; end;
if ~isfield(in,'obs'); in.obs = []; end;
if ~isfield(in,'ReturnOptimal'); in.ReturnOptimal = 1; end;

in.D = size(in.xmax,2); % dimensionality of inputs (search domain)

% the following should be provided by the user
%in.covfunc      = {@covSEard};       % GP kernel
%in.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use
%covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard,
%covRQard, respectively.
%in.hyp          = hyp;  % hyperparameters, with fields .lik and .cov
%in.xmin         = xmin; % lower bounds of rectangular search domain
%in.xmax         = xmax; % upper bounds of rectangular search domain
%in.MaxEval      = H;    % Horizon (number of evaluations allowed)
%in.f            = @(x) f(x) % handle to objective function

%% set up
GP              = struct;
GP.covfunc      = in.covfunc;
GP.covfunc_dx   = in.covfunc_dx;
%GP.covfunc_dx   = in.covfunc_dx;
%GP.covfunc_dxdz = in.covfunc_dxdz;
GP.likfunc      = in.likfunc;
GP.hyp          = in.hyp;
GP.res          = 1;
GP.deriv        = in.with_deriv;
GP.poly         = in.poly;
GP.log          = in.log;
%GP.SampleHypers = in.SampleHypers;
%GP.HyperSamples = in.HyperSamples;
%GP.HyperPrior   = in.HyperPrior;

GP.x            = in.x;
GP.y            = in.y;
GP.obs          = in.obs;
%GP.dy           = in.dy;
GP.K            = [];
GP.cK           = [];
GP.K   = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
GP.cK  = chol(GP.K);

GP.invL = inv(diag(exp(in.hyp.cov(1:end-1)))); %inverse of length scales

D = in.D;
S0= 0.5 * norm(in.xmax - in.xmin);

myparams = struct('S', 1000, 'Ny', 10, 'Nn', 8, 'Ntrial_s', 100, 'Neval', 20);

s_dim = 1; 
s_vec = linspace(in.xmin(1), in.xmax(1), myparams.Neval)';

if in.ReturnOptimal
    [xx, xy] = meshgrid(s_vec, linspace(in.xmin(2), in.xmax(2), 100)');
    
    out.val_opt = min(arrayfun(@(a,b)(in.f([a b])), xx, xy), [], 1)';
end

%% iterations
converged = false;
numiter   = 0;
MeanEsts  = zeros(0,D);
MAPEsts   = zeros(0,D);
BestGuesses= zeros(0,D);
while ~converged && (numiter < in.MaxEval)
    numiter = numiter + 1;
    fprintf('\n');
    disp(['iteration number ' num2str(numiter)])
%     try
        trial_contexts = in.xmin(1) + rand(myparams.Ntrial_s,1)*(in.xmax(1) - in.xmin(1));

        % sample belief and evaluation points
        %[zb,lmb]   = SampleBeliefLocations(GP,in.xmin,in.xmax,in.Nb,BestGuesses,in.PropFunc);
        %TODO should be Thompson sampling. This EI based is absolutely not
        %good
         zb = repmat(in.xmin, in.Nb,1) + rand(in.Nb, D).*repmat(in.xmax - in.xmin, in.Nb, 1);
         lmb = -log(norm([in.xmin' in.xmax']))*ones(in.Nb,1);  %log of uniform measure, |I|^-1
         [zt,sorti] = sort(zb(:,2));
         vals = [(zt(2)-zt(1))/2 + zt(1)-in.xmin(2);
                  (zt(3:end) - zt(1:end-2))/2;
                  in.xmax(2) - zt(end) + (zt(end)-zt(end-1))/2];
         vals(sorti) = vals; %restore order
         lmb = lmb + log(vals);
         %lmb = lmb + 
         %TODO this is not good, need better representers
         
        % generate zb, logP vectors
        zb_vec = zeros(in.Nb, in.D, myparams.Ntrial_s);
        lmb_vec = zeros(in.Nb, 1, myparams.Ntrial_s);
        logP_vec = zeros(in.Nb, 1, myparams.Ntrial_s);
        %Mb_vec = zeros(in.Nb, 1, myparams.Ntrial_s);
        %Vb_vec = zeros(in.Nb, in.Nb, myparams.Ntrial_s);
        for i=1:myparams.Ntrial_s
            zb_vec(:,:,i) = [repmat(trial_contexts(i,:),in.Nb,1) zb(:,s_dim+1:end)];
            lmb_vec(:,:,i) = lmb;
            %[Mb_vec(:,:,i),Vb_vec(:,:,i)]    = GP_moments(GP,zb_vec(:,:,i));
            logP_vec(:,:,i) = EstPmin(GP, zb_vec(:,:,i), myparams.S, randn(size(zb,1),myparams.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);          
        end
        
        % construct ACES function for this GP
        rand_start = rng();
        aces_f = @(x)(ACES(GP, logP_vec, zb_vec, lmb_vec, x, trial_contexts, myparams, rand_start));


%         % these are the locations where pmin will be approximated 
%         % uses expected improvement (EI) to select good points
%         % zb are the points, lmb is a measure according to EI. 
%         % In ACES paper the uniform measure Ui is lmb
%         
%         [Mb,Vb]    = GP_moments(GP,zb); % this was 3 time faster than gp function
%       
%         %[m my_s2] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, zb);
%         
%         % belief over the minimum on the sampled set
%         [logP,dlogPdM,dlogPdV,ddlogPdMdM] = joint_min(Mb,Vb);        % p(x=xmin)
%        % this is discrete distr p(x==xmin) using expectation propagation
%        
%         out.Hs(numiter) = - sum(exp(logP) .* (logP + lmb));       % current Entropy
%         
%         % store the best current guess as start point for later optimization.
%         [~,bli] = max(logP + lmb);
%         % is this far from all the best guesses? If so, then add it in.
%         ell = exp(GP.hyp.cov(1:D))';
%         if isempty(BestGuesses)
%             BestGuesses(1,:) = zb(bli,:);
%         else
%             dist = min(sqrt(sum(bsxfun(@minus,zb(bli,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D));
%             if dist > 0.1
%                 BestGuesses(size(BestGuesses,1)+1,:) = zb(bli,:);
%             end
%         end
%         % BestGuesses are used as initial points for optimization of GP min
%        % not used for selecting the next query point
%        
%         dH_fun     = dH_MC_local(zb,GP,logP,dlogPdM,dlogPdV,ddlogPdMdM,in.T,lmb,in.xmin,in.xmax,false,in.LossFunc);
%         dH_fun_p   = dH_MC_local(zb,GP,logP,dlogPdM,dlogPdV,ddlogPdMdM,in.T,lmb,in.xmin,in.xmax,true,in.LossFunc);
%         % sample some evaluation points. Start with the most likely min in zb.
%         [~,mi]     = max(logP);
%         xx         = zb(mi,:);
%         Xstart     = zeros(in.Ne,D);
%         Xend       = zeros(in.Ne,D);
%         Xdhi       = zeros(in.Ne,1);
%         Xdh        = zeros(in.Ne,1);
%         fprintf('\n sampling start points for search for optimal evaluation points\n')
%         xxs = zeros(10*in.Ne,D);
%         for i = 1:10 * in.Ne
%             if mod(i,10) == 1 && i > 1; xx = in.xmin + (in.xmax - in.xmin) .* rand(1,D); end;
%             xx     = Slice_ShrinkRank_nolog(xx,dH_fun_p,S0,true);
%             xxs(i,:) = xx;
%             if mod(i,10) == 0; Xstart(i/10,:) = xx; Xdhi(i/10) = dH_fun(xx); end
%         end
%         
% %         my_dh = @(x)(LossFunction(GP, logP, zb, lmb, x, [], myparams));
%        
% %          test = [];
% %         for i=1:in.Ne
% %             temp = [];
% %             for j=1:10
% %                 temp(:,j) = my_dh(Xstart(i,:));
% %             end
% %             test = [test; Xstart(i,:) dH_fun(Xstart(i,:)) mean(temp,2) temp std(temp)];
% %             
% %         end
% %         
        % optimize for each evaluation point:
%         fprintf('local optimizations of evaluation points\n')
%         for i = 1:in.Ne
%             [Xend(i,:),Xdh(i)] = fmincon(dH_fun,Xstart(i,:),[],[],[],[],in.xmin,in.xmax,[], ...
%                 optimset('MaxFunEvals',20,'TolX',eps,'Display','off','GradObj','on'));
%             
%         end
%         % which one is the best?
%         [xdhbest,xdhbv]   = min(Xdh);
        
        
        %% optiization with CMA-ES
%         fprintf('optimize with CMA-ES\n')
%         opts = cmaes('defaults');
%         opts.UBounds = in.xmax';
%         opts.LBounds = in.xmin';
%         opts.Restarts = 0;
%         opts.DispFinal = 'off';
%         opts.DispModulo = 'Inf';
%         opts.Noise.on = 0;  %1
%         opts.SaveVariables = 'on';
%         opts.Resume = 'no';
%         opts.MaxFunEvals = 100;
%         opts.PopSize = 10;
%         %opts.StopOnEqualFunctionValues = 100;
%         %opts.WarnOnEqualFunctionValues = 0;
%         %xstart = sprintf('%f + %f.*rand(%f,1)', theta_bounds(:,1), theta_bounds(:,2)-theta_bounds(:,1), size(theta_bounds,1)  );
%         %xstart = Xstart(xdhbv,:);
%         xstart = repmat(in.xmin, in.Ne,1) + rand(in.Ne, D).*repmat(in.xmax - in.xmin, in.Ne, 1);
%         insigma = (in.xmax-in.xmin)'/3;
%         
%         xatmin = zeros(D, in.Ne);
%         minval = zeros(1,in.Ne);
%         for i=1:in.Ne
%             [xatmin(:,i), minval(:,i), counteval, ~, cmaesout] = cmaes( ...
%             @(x)(aces_f(x')), ...    % name of objective/fitness function
%             xstart(i,:)', ...    % objective variables initial point, determines N
%             insigma, ...   % initial coordinate wise standard deviation(s)
%             opts ...    % options struct, see defopts below
%             );
%             counteval
%         end
%         [xatmin' minval']
%         [minval1, best] = min(minval);
%         xatmin1 = xatmin(:,best)';
        
        
        [minval1,xatmin1,hist] = Direct(struct('f', @(x)(aces_f(x'))), [in.xmin' in.xmax'], struct('showits', 1, 'maxevals', 40));
        xrange = [in.xmax' - in.xmin']/10;
        [minval2,xatmin2,hist] = Direct(struct('f', @(x)(aces_f(x'))), [xatmin1-xrange xatmin1+xrange], struct('showits', 1, 'maxevals', 40));
        
        %% print ACES function
        fprintf('plot ACES function\n')

        %for printing
        [xx, xy] = meshgrid(linspace(in.xmin(1),in.xmax(1),10)', linspace(in.xmin(2),in.xmax(2),10)');
      
        figure
        aces_values = arrayfun(@(a,b)(aces_f([a b])), xx, xy);
        mesh(xx, xy, aces_values);
        
        hold on;
        scatter3(GP.x(:,1), GP.x(:,2), in.f(GP.x), 'ro');
        %scatter3(xstart(:,1), xstart(:,2), arrayfun(@(a,b)aces_f([a b]), xstart(:,1), xstart(:,2)), 'b*');
        scatter3(xatmin1(1), xatmin1(2), minval1, 'y*');
        scatter3(xatmin2(1), xatmin2(2), minval2, 'r*');
        drawnow;
        

        %% plot pmin
        xx = linspace(in.xmin(1),in.xmax(1),100)';
        figure
        hold on
        for i=1:100
            zz = sort(zb(:,2));
            pmin_values(i,:) = EstPmin(GP, [repmat(xx(i),in.Nb,1) zz], 1000, randn(size(zb,1),1000));
            %%plot3(repmat(xx(i),1,in.Nb), pmin_values(i,:), zz');
        end
        
        [xx xy] = meshgrid(linspace(in.xmin(1),in.xmax(1),100)', zz);

        mesh(xx, xy, exp(pmin_values)');
        

        %% eval function
        fprintf('evaluating function \n')
        %xp                = Xend(xdhbv,:);
        xp = xatmin2';
        yp                = in.f(xp);
        
        GP.x              = [GP.x ; xp ];
        GP.y              = [GP.y ; yp ];
        %GP.dy             = [GP.dy; dyp];
        GP.K              = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
        GP.cK             = chol(GP.K);
        
        %% estimate minimum
%         MeanEsts(numiter,:) = sum(bsxfun(@times,zb,exp(logP)),1);
%         [~,MAPi]            = max(logP + lmb);
%         MAPEsts(numiter,:)  = zb(MAPi,:);
%         
%         fprintf('finding current best guess\n')
%         [out.FunEst(numiter,:),FunVEst] = FindGlobalGPMinimum(BestGuesses,GP,in.xmin,in.xmax);
%         % is the new point very close to one of the best guesses?
%         [cv,ci] = min(sum(bsxfun(@minus,out.FunEst(numiter,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D);
%         if cv < 2.5e-1 % yes. Replace it with this improved guess
%             BestGuesses(ci,:)  = out.FunEst(numiter,:);
%         else % no. Add it to the best guesses
%             BestGuesses(size(BestGuesses,1)+1,:) = out.FunEst(numiter,:);
%         end
    
        
        %% evaluate over contexts
        fprintf('evaluate over contexts\n')
        [xx, xy] = meshgrid(linspace(in.xmin(1),in.xmax(1),50)', linspace(in.xmin(2),in.xmax(2),50)');
        theta_vec = [];
        
        for i=1:length(s_vec)
            [theta, val] = ACESpolicy(GP, s_vec(i,:), [in.xmin(2)' in.xmax(2)']);    
            theta_vec(i, :) = theta;
        end
        val_vec = in.f([s_vec theta_vec]);
        current_performance = sum(val_vec)
        out.val_vec(:,numiter) = val_vec;
        
        figure
        real_val = arrayfun(@(a,b)(in.f([a b])), xx, xy);
        mesh(xx, xy, real_val)
        
        hold on;
        scatter3(GP.x(:,1), GP.x(:,2), in.f(GP.x), 'ro');
        scatter3(s_vec(:,1), theta_vec(:,1), val_vec, 'y*');
        %scatter3(out.FunEst(numiter,1), out.FunEst(numiter,2), in.f(out.FunEst(numiter,:)), 'r*');    

        figure
        [my_m my_s2] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, [xx(:) xy(:)]);
         mesh(xx,xy, reshape(my_m, size(xx)));
        hold on;
        scatter3(GP.x(:,1), GP.x(:,2), in.f(GP.x), 'ro');
        [my_m my_s2] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, [s_vec theta_vec]);
        scatter3(s_vec(:,1), theta_vec(:,1), my_m, 'y*');
        %scatter3(out.FunEst(numiter,1), out.FunEst(numiter,2), in.f(out.FunEst(numiter,:)), 'r*');    
                
        drawnow;
        
        %% optimize hyperparameters
        if in.LearnHypers
            minimizeopts.length    = 10;
            minimizeopts.verbosity = 1;
            GP.hyp = minimize(GP.hyp,@(x)in.HyperPrior(x,GP.x,GP.y),minimizeopts);
            GP.K   = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
            GP.cK  = chol(GP.K);
            fprintf 'hyperparameters optimized.'
            display(['length scales: ', num2str(exp(GP.hyp.cov(1:end-1)'))]);
            display([' signal stdev: ', num2str(exp(GP.hyp.cov(end)))]);
            display([' noise stddev: ', num2str(exp(GP.hyp.lik))]);
        end
        
        out.GPs{numiter} = GP;
%     catch error
%         if numiter > 1
%             out.FunEst(numiter,:) = out.FunEst(numiter-1,:);
%         end
%         fprintf('error occured. evaluating function at random location \n')
%         xp                = in.xmin + (in.xmax - in.xmin) .* rand(1,D);
%         yp                = in.f(xp);
%         
%         GP.x              = [GP.x ; xp ];
%         GP.y              = [GP.y ; yp ];
%         %GP.dy             = [GP.dy; dyp];
%         GP.K              = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
%         GP.cK             = chol(GP.K);
%         
%         MeanEsts(numiter,:) = sum(bsxfun(@times,zb,exp(logP)),1);
%         [~,MAPi]            = max(logP + lmb);
%         MAPEsts(numiter,:)  = zb(MAPi,:);
%         
%         out.errors{numiter} = error;
%     end
end

%% construct output
out.GP       = GP;
%out.MeanEsts = MeanEsts;
%out.MAPEsts  = MAPEsts;
%out.logP     = logP;
