function [ stats, linstat, params ] = MyEntropySearch(input_params)
% TODO: try preserving st context correlation in policy prediction:
% sample a couple of representer st from covariance function around 
% query context and predict from that full GP
% This only makes sense if we hope to get something out of this assumption
% it does bad if there are big discountunities between nearby context

% TODO: representer sampling not great yet (Thompson?) and it is not
% compensated in the Loss function (not added to lmb). Due to this entropy
% change will be large where lots of points are around the new sample but
% will be little where there are few points around new sample. This is
% totally wrong.
% Try finding out how they did Thompson, or try using EI as in ES paper

% there are peaks on the bounds now on the weighted pmin. it shouldnt be because of non-uniform sampling
% problem might be with GP hyperparameters? assumes two strong correlation
% along theta: so the conditional posterior on a context is very smooth:
% always gives bound as minimum. with lower corrleation there should be
% more wiggling 

% probabilistic line search algorithm that adapts it search space
% stochastically, by sampling search points from their marginal probability of
% being smaller than the current best function guess.
%
% (C) Philipp Hennig & Christian Schuler, August 2011

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

params = struct(...
    'problem', ToyBall, ...
    'Rcoeff', [], ...
    'RandomiseProblem', true, ...
    'GP', struct, ... %user may override this with initial samples, etc
    ...
    'S', 1000, ... %how many samples to take from GP posterior to estimate pmin
    'Ny', 10, ... %how many samples to predict pmin given a new x
    'Ntrial_st', 20, ...  %representers for st space, can be number or vector
    'Ntrial_se', 20, ... %representers for se space, can be number or vector
    'Nn', 8, ...  % only the neares Nn out of Ntrial_se will be evaluated for a context se
    'Nb', 30, ... %number of representers for p_min over theta space generated with Thompson sampling
    'Nbpool', 500, ... %randomply chosen theta value pool for Thompson sampling
    'Neval', [8, 8, 20, 20, 20, 20, 20, 20, 20, 20, 20], ... %evaluation points over contexts. Theta space will be used computing optimal values
    'DirectEvals1', 100, ...  % number of maximum function evaluations for DIRECT search
    'DirectEvals2', 100, ...
    'DirectIters1', 10, ...  % number of maximum iterations for DIRECT search
    'DirectIters2', 10, ...
    ... % GP parameters. only used if GP is not provided. covarianve values
    'sigmaM0', [], ... %[0.01; 0.01],... % lengthscale (std, not cov), how much inputs should be similar in that dim.
    ...               % i.e. how far inputs should influence each other
    ...               % can be single value or vector for each theta dim
    'sigmaF0', [], ... %0.8,...  % how much inputs are correlated - (std, not cov)
    'sigma0', [], ... % noise level on signals (std, not cov);
    'Normalize', 0, ... %normalize y values: offse
    'OptimisticMean', 0.5, ... %lowest possible value (will shift y values)
    ... %TODO these are not normalized!
    'Algorithm', 1, ...   % 1 ACES, 2, BOCPSEntropy, 3 Active-BOCPS, 4 BOCPS with direct+direct (set Ntrial_st=1) 5 BOCPS with that optimization method
    'Sampling', 'Thompson3', ...  %can be Thompson, Nothing, Thompson2 Thompson3 None
    'kappa', 0.5, ... % kappa for BOCPS acquisition function
    ...
    'LearnHypers', false, ...
    'HyperPrior',@SEGammaHyperPosterior,... %for learning hyperparameters
    'Niter', 40, ...
    'InitialSamples', 9, ...  %minimum 1
    'EvalModulo', 100, ...
    'EvalAllTheta', 0, ...
    'ReturnOptimal', 0, ... %computes optimal values and put in return struct
    'ConvergedFunc', @()(false), ... %this will be called at end of iteration
    'output_off', 0);

PlotModulo = struct('ACES', 0, 'pmin', 0, 'policy', 0, 'real', 0);

if (exist('input_params'))
    params = ProcessParams(params, input_params);
end


%% obsolate entropy search variables
%if ~isfield(params,'likfunc'); params.likfunc = @likGauss; end; % noise type
%if ~isfield(params,'poly'); params.poly = -1; end; % polynomial mean?
%if ~isfield(params,'log'); params.log = 0; end;  % logarithmic transformed observations?
%if ~isfield(params,'with_deriv'); params.with_deriv = 0; end; % derivative observations?
%if ~isfield(params,'x'); params.x = []; end;  % prior observation locations
%if ~isfield(params,'y'); params.y = []; end;  % prior observation values
%if ~isfield(params,'T'); params.T = 200; end; % number of samples in entropy prediction
%if ~isfield(params,'Ne'); params.Ne = 4; end; % number of restart points for search
%if ~isfield(params,'LossFunc'); params.LossFunc = {@LogLoss}; end;
%if ~isfield(params,'PropFunc'); params.PropFunc = {@EI_fun}; end;

%% setup entropy search specific parameters that depend on the problem
problem = params.problem;
params.problem = struct; %clear so dont have to pass in to parfor later
if params.RandomiseProblem
    problem.Randomise();
end
if ~isempty(params.Rcoeff)
    problem.SetRcoeff(params.Rcoeff);
end

% set problem specific default GP hyperparams
if isempty(params.sigmaM0) 
    params.sigmaM0 = problem.def_sigmaM0;
end
if isempty(params.sigmaF0) 
    params.sigmaF0 = problem.def_sigmaF0;
end
if isempty(params.sigma0) 
    params.sigma0 = problem.def_sigma0;
end

params.xmin = [problem.st_bounds(:,1)' problem.se_bounds(:,1)' problem.theta_bounds(:,1)'];
params.xmax = [problem.st_bounds(:,2)' problem.se_bounds(:,2)' problem.theta_bounds(:,2)'];
params.D = size(params.xmax,2); % dimensionality of inputs (search domain)
params.Neval = params.Neval(1:params.D);

plot_x = (params.xmax - params.xmin)*1 + params.xmin; %this will be used when not plotting specific dimension

% dimensionality helper variables
D = params.D;
st_dim = size(problem.st_bounds,1);
se_dim = size(problem.se_bounds,1);
params.st_dim = st_dim;
params.se_dim = se_dim;
s_dim = params.st_dim + params.se_dim;
th_dim = D-s_dim;
sti = 1:st_dim;
sei = st_dim+1:st_dim+se_dim;
thi = s_dim+1:D;

if size(params.sigmaM0,1) == 1
    params.sigmaM0 = repmat(params.sigmaM0, se_dim+th_dim, 1);
end
if ~st_dim || params.Algorithm == 4 || params.Algorithm == 5
    params.Ntrial_st = 1;
end
if ~se_dim || params.Algorithm == 4 || params.Algorithm == 5
    params.Ntrial_se = 1;
end
params.Nn = min(params.Nn, params.Ntrial_se);
if params.Algorithm == 3 || params.Algorithm == 4 || params.Algorithm == 5
    params.Sampling = 'None';
end

%% setup default GP and override with input if provided
GP              = struct;
GP.covfunc      = {@covSEard};       % GP kernel
GP.covfunc_dx   = {@covSEard_dx_MD}; % derivative of GP kernel. You can use covSEard_dx_MD and covRQard_dx_MD if you use Carl's & Hannes' covSEard, covRQard, respectively.
GP.likfunc      = {@likGauss};
hyp = struct;
hyp.cov         = log([params.sigmaM0; params.sigmaF0]); % hyperparameters for the kernel
hyp.lik         = log(params.sigma0); % noise level on signals (log(standard deviation));
GP.hyp          = hyp;
GP.res          = 1;
GP.offset       = -params.OptimisticMean;
GP.deriv        = 0; %from entropy search ? derivative observations?
GP.poly         = -1; %from entropy search ? polynomial mean?
GP.log          = 0; % logarithmic transformed observations?
GP.x            = samplerange(params.xmin([sei thi]), params.xmax([sei thi]), params.InitialSamples);
[GP.y, GP.obs]  = problem.sim_func([repmat(plot_x(sti),params.InitialSamples,1) GP.x]);
%GP.covfunc_dx   = in.covfunc_dx;
%GP.covfunc_dxdz = in.covfunc_dxdz;
%GP.SampleHypers = in.SampleHypers;
%GP.HyperSamples = in.HyperSamples;
%GP.HyperPrior   = in.HyperPrior;
%GP.dy           = in.dy;

if (isfield(params,'GP'))
    GP = ProcessParams(GP, params.GP);
    params.InitialSamples = size(GP.x, 1);
end

GP.invL = inv(diag(exp(GP.hyp.cov(1:end-1)))); %inverse of length scales
GP.K   = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
GP.cK  = robustchol(GP.K);

GP.hyp_initial = GP.hyp;

prior.cov = {};
for i = 1:size(GP.hyp.cov,1)
   prior.cov{i} = {@priorTransform, @exp, @exp, @log, {@priorGauss, exp(GP.hyp_initial.cov(i)), exp(GP.hyp_initial.cov(i))/4}};
end
prior.lik = {{@priorClamped}};
prior.lik = {@priorTransform, @exp, @exp, @log, {@priorGauss, exp(GP.hyp_initial.lik), exp(GP.hyp_initial.lik)/4}};

GP.inf = {@infPrior, @infExact, prior};

% optimize hyper parameters if needed
GP = problem.MapGP(GP, [], params.LearnHypers);

if params.output_off
    PlotModulo = struct('ACES', 0, 'pmin', 0, 'policy', 0);
end
if strcmp(params.Sampling, 'Nothing')
    params.Nb = params.Nbpool;
end

stats = struct('last_R_mean', 0, 'R_opt', 0);
linstat = struct('R_mean', zeros(params.InitialSamples,1), ...
                 'st', zeros(size(GP.x,1),1), 'se', GP.x(:,sei-st_dim), 'theta', GP.x(:,thi-st_dim), ...
                 'theta_s', zeros(params.InitialSamples, prod(params.Neval([sti sei]))*th_dim), ...
                 'R_s', zeros(params.InitialSamples, prod(params.Neval([sti sei]))), ...
                 'R_opt', [], ...
                 'outcome', GP.obs(:,:), 'evaluated', zeros(params.InitialSamples,1), ...
                 'GP', repmat(GP.hyp, params.InitialSamples, 1));

%% iterations
if ~params.output_off
    fprintf 'starting entropy search.\n'
end
converged = false;
numiter   = params.InitialSamples;
%MeanEsts  = zeros(0,D);
%MAPEsts   = zeros(0,D);
%BestGuesses= zeros(0,D);
while ~converged && (numiter < params.Niter)
    numiter = numiter + 1;
    if ~params.output_off
        fprintf('\n');
        disp(['iteration number ' num2str(numiter)])
    end
        
    %% Build context representers
    st_trials = samplerange(params.xmin(sti), params.xmax(sti), params.Ntrial_st);
    se_trials = samplerange(params.xmin(sei), params.xmax(sei), params.Ntrial_se);
    % SORT does not make sense for higher dim!
    
    %for Algorithm 2 and 4, 5 need to sample random context 
    if params.Algorithm == 2 || params.Algorithm == 4 || params.Algorithm == 5
        %context = samplerange(params.xmin([sti sei]), params.xmax([sti sei]), 1);
        context = zeros(1,st_dim+se_dim);
        if ~isempty(sti)
            context(sti) = st_trials(1,:);
            % st part of context is not used anyway
        end
        if ~isempty(sei)
            context(sei) = se_trials(1,:);
        end
        % enough to compute ACES for nearest Nn to random se
        if se_dim
            dm = zeros(size(se_trials,1),1);
            for i=1:size(se_trials,1)
                dm(i) = mahaldist2(se_trials(i,:), context(sei), GP.invL(sei, sei));
            end
            [sortedX, sortedIndices] = sort(dm,'ascend');
            rel_se_inds = sortedIndices(1:params.Nn);
            se_trials = se_trials(rel_se_inds,:);
        end
    end
    
    zb = samplerange(params.xmin(thi), params.xmax(thi), params.Nbpool);
    lmb = log(norm(params.xmax([sei thi]) - params.xmin([sei thi])));
    logP = zeros(params.Nbpool, 1); %just so plot doesnt fail when unused
    % log(params.Nb); %= -log(1/Nb) %log of uniform measure, |I|^-1
    
    % generate zb, logP vectors for each trial se, st pairs
    zb_vec = zeros(params.Nb, se_dim+th_dim, size(se_trials,1), params.Ntrial_st);
    lmb_vec = zeros(params.Nb, 1, size(se_trials,1), params.Ntrial_st);
    logP_vec = zeros(params.Nb, 1, size(se_trials,1), params.Ntrial_st);
    offset_vec = zeros(1, 1, size(se_trials,1), params.Ntrial_st); % used to offset BOCPS acquisition function with best predicted reward at (st,se)
    GP_cell = cell(params.Ntrial_st, 1);
    
    zb_vec2 = zeros(params.Nbpool, se_dim+th_dim, size(se_trials,1), params.Ntrial_st);
    lmb_vec2 = zeros(params.Nbpool, 1, size(se_trials,1), params.Ntrial_st);
    logP_vec2 = zeros(params.Nbpool, 1, size(se_trials,1), params.Ntrial_st);

    % construct GP_cell for each st
    for i_st=1:params.Ntrial_st
        GP_cell{i_st} = problem.MapGP(GP, st_trials(i_st,:), params.LearnHypers);
    end
    
    % construct zb representers for se+th 
    for i_st=1:params.Ntrial_st
        for i_se=1:size(se_trials,1)
            if (params.Algorithm == 3)
                 acqfun = @(theta)(gp(GP_cell{i_st}.hyp, [], [], GP_cell{i_st}.covfunc, GP_cell{i_st}.likfunc, GP_cell{i_st}.x, GP_cell{i_st}.y, [se_trials(i_se,:) theta']));
                 [minval1,~, ~] = Direct(struct('f', acqfun), [params.xmin(thi)' params.xmax(thi)'], struct('showits', 0));
                 offset_vec(1,1,i_se,i_st) = minval1;
                 continue;
            end
%             zb_vec(:,:,i_se, i_st) = [repmat(se_trials(i_se,:),params.Nb,1) zb];
%             lmb_vec(:,:,i_se, i_st) = lmb;
%             logP_vec(:,:,i_se, i_st) = EstPmin(GP_cell{i_st}, zb_vec(:,:,i_se, i_st), params.S, randn(size(zb,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);

%             % Sample from current Pmin ==? Thompson sampling
%             zbnew = zeros(params.Nb, se_dim+th_dim);
%             for i=1:params.Nb
%                 pool = samplerange(params.xmin(thi), params.xmax(thi), params.Nbpool);
%                 extpool = [repmat(se_trials(i_se,:),size(pool,1),1) pool];
%                 [sample ind] = SamplePmin(GP_cell{i_st}, extpool);
%                 zbnew(i, :) = sample;
%             end
%             [~, ind] = sort(zbnew(:,end));
%             zbnew = zbnew(ind, :);

            % % Sample from current Pmin more efficiently (approx)
            %pool = samplerange(params.xmin(thi), params.xmax(thi), params.Nbpool);
            if strcmp(params.Sampling, 'Thompson')
                pool = zb;
                extpool = [repmat(se_trials(i_se,:),size(pool,1),1) pool];
                [sample ind] = SamplePmin(GP_cell{i_st}, extpool, params.Nb);
                zbnew(:, :) = sample;
                [~, ind] = sort(zbnew(:,end));
                zbnew = zbnew(ind, :);
            
%             % compute weights (u in paper) with nearest neighbors
%             % it should be the actual Pmin, but we cant really represent
%             % that, exactly thats why we are trying better sampling 
%             % locations..
%             nearest_ids = knnsearch(zb, zbnew(:,thi));
%             w = logP(nearest_ids);
%             % this is inaccurate: because nearest value fluctuates a lot
%             % with uniform samples its bad to estimate + for 500 locations
%             % we sample only 1000 outcomes..
%             
%             % option: use K nearest neighboor 
            
            % alternativy: explicitly compute widths and use inverse as weights
            % this wont work for higher dim
                zt = zbnew(:,end); %theta line
                vals = [(zt(2)-zt(1))/2 + zt(1)-params.xmin(end);
                      (zt(3:end) - zt(1:end-2))/2;
                       params.xmax(end) - zt(end) + (zt(end)-zt(end-1))/2];
                lw = -(log(vals) - logsumexp(log(vals))) ; % no need for logsumexp this is already sums to 1

                lmb_vec(:,:,i_se, i_st) = lmb + lw; % + log(params.Nb);
                zb_vec(:,:,i_se, i_st) = zbnew;
                logP_vec(:,:,i_se, i_st) = EstPmin(GP_cell{i_st}, zbnew, params.S, randn(size(zbnew,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);
            elseif strcmp(params.Sampling, 'Thompson2')
                pool = zb;
                extpool = [repmat(se_trials(i_se,:),size(pool,1),1) pool];
                logP = EstPmin(GP_cell{i_st}, extpool, params.Nbpool*25, randn(params.Nbpool, params.Nbpool*25));
                P = exp(logP);
                % sample from logP with replacement. w indicuates number of samples 
                w = zeros(params.Nbpool,1);
                inds = zeros(params.Nb, 1);
                i = 1;
                while i<=params.Nb
                    y = randsample(params.Nbpool,1,true,P);
                    if w(y) > 0
                        %TODO problem: can stuck in infinite loop
                        w(y) = w(y)+1;
                    else
                        w(y) = 1;
                        inds(i) = y;
                        i = i+1;
                    end
                end
                % sort for easier debugging, not neccessary
                inds = sort(inds);                
                
                w = w(inds,:);

                zbnew = extpool(inds, :);
                lw = logP(inds,:) - log(w);  %log(u/w)
                
                lmb_vec(:,:,i_se, i_st) = lmb + lw; % + log(params.Nb);
                zb_vec(:,:,i_se, i_st) = zbnew;
                logP_vec(:,:,i_se, i_st) = EstPmin(GP_cell{i_st}, zbnew, params.S, randn(size(zbnew,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);

                x=1;
                %when sampling new logP on these representers P*1/w should
                %be unifrom, and P*u/w should resemble real PDF
                
                zb_vec2(:,:,i_se, i_st) = extpool;
                logP_vec2(:,:,i_se, i_st) = logP;
                lmb_vec2(:,:,i_se, i_st) = lmb + log(params.Nbpool) - lmb; %because u=1, Zu = xmax-xmin. b(x) = 1/xmax-xmin
         
            elseif strcmp(params.Sampling, 'Thompson3')
                pool = zb;
                extpool = [repmat(se_trials(i_se,:),size(pool,1),1) pool];
                logP = EstPmin(GP_cell{i_st}, extpool, params.Nbpool*25, randn(params.Nbpool, params.Nbpool*25));
                P = exp(logP);
                % sample from logP without replacement
                inds = datasample((1:params.Nbpool)',params.Nb,1,'Replace',false, 'Weights', P);
                % the probability that a zb was choosen is actually
                % Multivariate Wallenius' noncentral hypergeometric distribution
                % where: n=20; N=500; all mi = 1; x=1; wi = Pi;
                
                
                % sort for easier debugging, not neccessary
                inds = sort(inds);
                
                zbnew = extpool(inds, :);
                lw = logP(inds,:);  %log(u)
                
                lmb_vec(:,:,i_se, i_st) = lmb + lw; % + log(params.Nb);
                zb_vec(:,:,i_se, i_st) = zbnew;
                logP_vec(:,:,i_se, i_st) = EstPmin(GP_cell{i_st}, zbnew, params.S, randn(size(zbnew,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);

                if(false)
                    [exp(logP(inds,:)) w]
                    figure
                    plot(extpool(:,end), exp(logP))
                    hold on
                    
                    logPthis = EstPmin(GP_cell{i_st}, zbnew, params.S, randn(size(zbnew,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);
                    correctedP = exp(logPthis)./w;
                    scatter(zbnew(:,end), correctedP);
                    
                   % steps = exp(-lw+logsumexp(lw)-log(sum(w)))*(params.xmax(end)-params.xmin(end));
                   % for i=1:params.Nb
                   %     line([zbnew(i,end)-steps(i)/2 zbnew(i,end)+steps(i)/2], [correctedP(i)  correctedP(i)]);
                   % end
                    
                    heights = exp(logPthis + lw); % + log(params.Nb)
                    % rescale hieghts to be comparable with other curve. but
                    % real pmin(x) may have very high values: note that not sum
                    % but integral has to sum to 1
                    heights = heights / sum(heights);
                    scatter(zbnew(:,end), heights , 'b');
                end
            elseif strcmp(params.Sampling, 'None')
                % do nothing
                %logP = zeros(params.Nbpool, 1);
            else
                zb_rel = [repmat(se_trials(i_se,:),size(zb,1),1) zb];
                zb_vec(:,:,i_se, i_st) = zb_rel;
                logP_vec(:,:,i_se, i_st) = EstPmin(GP_cell{i_st}, zb_rel, params.S, randn(size(zb_rel,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);
                lmb_vec(:,:,i_se, i_st) = lmb + log(params.Nb) - lmb; %because u=1, Zu = xmax-xmin. b(x) = 1/xmax-xmin
                
            end
            
            if (false) % for debugging
                figure
                zb_rel = [repmat(se_trials(i_se,:),size(zb,1),1) zb];
                logP = EstPmin(GP_cell{i_st}, zb_rel, params.S, randn(size(zb,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);
                plot(zb_rel(:,end), exp(logP))
                hold on
                
                logPthis = EstPmin(GP_cell{i_st}, zbnew, params.S, randn(size(zbnew,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);
                scatter(zbnew(:,end), exp(logPthis));
                heights = exp(logPthis + w); % + log(params.Nb)
                % rescale hieghts to be comparable with other curve. but
                % real pmin(x) may have very high values: note that not sum
                % but integral has to sum to 1
                heights = heights / sum(heights);
                
                scatter(zbnew(:,end), heights , 'b');
                steps = exp(-w)*(params.xmax(end)-params.xmin(end));
                for i=1:params.Nb
                    line([zbnew(i,end)-steps(i)/2 zbnew(i,end)+steps(i)/2], [heights(i)  heights(i)]);
                end
                
                [Mb Vb] = gp(GP_cell{i_st}.hyp, [], [], GP_cell{i_st}.covfunc, GP_cell{i_st}.likfunc, GP_cell{i_st}.x, GP_cell{i_st}.y, ...
                zb_rel);
                plot(zb(:,end), Mb, 'r')
                plot(zb(:,end), Mb + 2*Vb, 'g-')
                
                %scatter(zb_rel(sel,end), exp(logP(sel)))
                %figure
                %scatter(zb_rel(sel,end), u(sel))
            end
            
%            % Beta distribution strategy
%            zb_rel = [repmat(se_trials(i_se,:),size(zb,1),1) zb];
%            logP = EstPmin(GP_cell{i_st}, zb_rel, params.S, randn(size(zb,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);
%                        
%             alpha = 200*exp(logP);
%             beta = 200-alpha;
%             u = betarnd(alpha+1, beta+1);
%             [~, ind] = sort(u, 'descend');
%             sel = sort(ind(1:params.Nb)); %select first Nb
%             zb_vec(:,:,i_se, i_st) = zb_rel(sel, :);
%             % we sampled proportional to the current logP
%             % more precisely, the probabilites are proportional to the Beta distr. mean
%             % this was not good: rescaling is not accurate due to P(x)=0-s.
%             logP_vec(:,:,i_se, i_st) = EstPmin(GP_cell{i_st}, zb_vec(:,:,i_se, i_st), params.S, randn(params.Nb, params.S));
%             lmb_vec(:,:,i_se, i_st) = lmb; %+ logPsel(sel) - logsumexp(logPsel(sel));% + log(u(sel));
%             if th_dim == 1
%                 %note: need to be sorted if not already (1+ dim)
%                 zt = zb_vec(:,end,i_se, i_st); %theta line
%                 vals = [(zt(2)-zt(1))/2 + zt(1)-params.xmin(end);
%                     (zt(3:end) - zt(1:end-2))/2;
%                     params.xmax(end) - zt(end) + (zt(end)-zt(end-1))/2];
%                 %vals(sorti) = vals; %restore order
%                 lmb_vec(:,:,i_se, i_st) = lmb_vec(:,:,i_se, i_st) - (log(vals) - logsumexp(log(vals) - log(params.Nb)));
%                 %TODO this is not good, need better representers
%             else
%                 disp('Note: for theta_dim > 1 this is not great\n');
%             end
%             %% recompute weights strategy
%             %alpha = 200*exp(logP_vec(:,:,i_se, i_st));
%             %beta = 200-alpha;
%             %logPsel2 = -log((1+(beta+1)./(alpha+1)));   %1/(1+b/a)
%             %[-(log(vals) - logsumexp(log(vals))), -logPsel - logsumexp(-logPsel) -logPsel2 - logsumexp(-logPsel2)]

        end
        
    end

    %% ACES function
     
    % construct ACES function for this GP
    if params.Algorithm == 1 || params.Algorithm == 3
        % there is no random context, also has to search for SE space
        context = [];
        ssei = sei;
        sinvsei = [];
    else
        % dont search in SE space
        ssei = [];
        sinvsei = sei;
    end
    
    % choose acquisition function to optimize
    if params.Algorithm == 1 || params.Algorithm == 2
        
        rand_start = rng();
        % rand_start = rng('shuffle');
        %aces_f = @(x)(ACES(GP, logP_vec, zb_vec, lmb_vec, y_vec, x, trial_contexts, in.st_dim, params, rand_start));
        %aces_f = @(x)(ACES2(GP, zb, lmb, x, in.st_dim, problem.MapGP, params, rand_start));
        aces_f = @(x)(ACES3(GP_cell, logP_vec, zb_vec, lmb_vec, st_trials, se_trials, [context(sinvsei) x], params, rand_start));
        %aces_f2 = @(x)(ACES3(GP_cell, logP_vec2, zb_vec2, lmb_vec2, st_trials, se_trials, [context(sinvsei) x], params, rand_start));
    elseif params.Algorithm == 3 || params.Algorithm == 4
        aces_f = @(x)(acq_func_bo3(GP_cell, offset_vec, st_trials, se_trials, [context(sinvsei) x], params));
    elseif params.Algorithm == 5
        aces_f = @(x)(acq_func_bo2(GP_cell{1}, [context(sinvsei), x], params.kappa));
    end
        
    % optimize 
    if params.Algorithm == 1 || params.Algorithm == 2 || params.Algorithm == 3 ||  params.Algorithm == 4
        [minval1,xatmin1,hist] = Direct(struct('f', @(x)(aces_f(x'))), [params.xmin([ssei thi])' params.xmax([ssei thi])'], struct('showits', (params.output_off == 0), 'maxits', params.DirectIters1, 'maxevals', params.DirectEvals1));
        if params.DirectIters2
            xrange = [params.xmax([ssei thi])' - params.xmin([ssei thi])']/10;
            xrange = [max(xatmin1-xrange,params.xmin([ssei thi])') min(xatmin1+xrange,params.xmax([ssei thi])') ];
            [minval2,xatmin2,hist] = Direct(struct('f', @(x)(aces_f(x'))), xrange, struct('showits', (params.output_off == 0), 'maxits', params.DirectIters2, 'maxevals', params.DirectEvals2));
        else
            minval2 = minval1;
            xatmin2 = xatmin1;
        end
       
    elseif  params.Algorithm == 5
        % RBOCPS
        [xatmin2 minval2] = ACESpolicy(@(x)(aces_f(x')), [params.xmin([ssei thi])' params.xmax([ssei thi])']);
        xatmin2 = xatmin2';
        xatmin1 = xatmin2; minval1 = minval2;
    end
    
    % add random context if algorithm 2, 4, or 5
    xatmin1 = [context(sinvsei)'; xatmin1];
    xatmin2 = [context(sinvsei)'; xatmin2];
    if ~params.output_off
        xatmin2
    end
    
    GP_full_x = repmat(plot_x, size(GP.x,1), 1);
    GP_full_x(:,[sei thi]) = GP.x;
    
    %% plot ACES function
    if PlotModulo.ACES && ~mod(numiter, PlotModulo.ACES)
        fprintf('plot ACES function\n')
        if se_dim+th_dim < 2
            [~, ind] = sort(st_trials(:,end));
            [xx, xy] = ndgrid(st_trials(ind,end), linspace(params.xmin(end),params.xmax(end),50)');
            aces_values = zeros(size(xx));
            for i=1:size(xx,1)
                aces_fplot = @(x)(ACES3(GP_cell(ind(i)), logP_vec(:,:,:,ind(i)), zb_vec(:,:,:,ind(i)), lmb_vec(:,:,:,ind(i)), st_trials(ind(i),:), se_trials, x, params, rand_start));
                aces_values(i,:) = arrayfun(@(b)(aces_fplot([plot_x(1:end-2), b])), xy(i,:));
            end
            
        elseif params.Algorithm == 2 || params.Algorithm == 4 || params.Algorithm == 5
            [xx, xy] = ndgrid(linspace(params.xmin(end-1),params.xmax(end-1),10)', linspace(params.xmin(end),params.xmax(end),10)');
            aces_values = arrayfun(@(a,b)(aces_f([plot_x(st_dim+se_dim+1:end-2) a(th_dim>1) b])), xx, xy);
        else
            [xx, xy] = ndgrid(linspace(params.xmin(end-1),params.xmax(end-1),10)', linspace(params.xmin(end),params.xmax(end),10)');
            aces_values = arrayfun(@(a,b)(aces_f([plot_x(st_dim+1:end-2) a b])), xx, xy);
        end
        figure
        mesh(xx, xy, aces_values);
        
        hold on;
        scatter3(GP_full_x(:,end-1), GP_full_x(:,end), ones(size(GP.x,1),1)*max(max(aces_values)), 'ro');
        xatmin1full = plot_x;
        xatmin1full([sei thi]) = xatmin1;
        xatmin2full = plot_x;
        xatmin2full([sei thi]) = xatmin2;
        scatter3(xatmin1full(end-1), xatmin1full(end), max(minval1, min(min(aces_values))), 'b*');
        scatter3(xatmin2full(end-1), xatmin2full(end), max(minval2, min(min(aces_values))), 'r*');
        %scatter3(xstart(:,1), xstart(:,2), arrayfun(@(a,b)aces_f([a b]), xstart(:,1), xstart(:,2)), 'b*');
        
        %plot current entropy over contexts
        if se_dim == 1
            Hvec = zeros(size(se_trials,1),1);
            for i=1:size(se_trials,1)
                Hvec(i) = - sum(exp(logP_vec(:,:,i,1)) .* (logP_vec(:,:,i,1) + lmb));           % current Entropy 
            end
            figure
            plot(se_trials(:,end), Hvec, '-*');
            hold on
            scatter(GP.x(:,sei(end)), ones(size(GP.x,1),1)*mean(Hvec))
            
        end

        
        drawnow;
    end
    
    %% plot pmin
    % only 1D theta supported for now
    if PlotModulo.pmin && ~mod(numiter, PlotModulo.pmin)
        [xx xy] = ndgrid(linspace(params.xmin(end-1),params.xmax(end-1),100)', linspace(params.xmin(end),params.xmax(end),100)');

        if se_dim + th_dim < 2
            %plot over a uniform grid
            pmin_values = zeros(size(xx));

            for i=1:size(xx,1)
            
                GPrel = problem.MapGP(GP, [plot_x(1:st_dim-1) xx(i,1)], params.LearnHypers);
                pmin_values(i,:) = EstPmin(GPrel, [xy(i,:)'], 1000, randn(size(xy,2),1000));
            end
            %%plot3(repmat(xx(i),1,in.Nb), pmin_values(i,:), zz');
        elseif th_dim == 2
            
            
            %plot zb pool and the selected representers for the closest GP
            ind = min(10, params.Ntrial_st);
            figure
            hold on
            stem3(zb(:,1), zb(:,2), exp(logP), 'bo');
            stem3(zb_vec(:,1,:,ind), zb_vec(:,2,:,ind), exp(logP_vec(:,:,:,ind)), 'r*');
            
            dist = zeros(size(xx(:),1),size(zb,1));
            for i=1:size(dist,1)
                dist(i,:) = (zb(:,1)'-xx(i)).^2 + (zb(:,2)'-xy(i)).^2;
            end
            [~, ind] = min(dist, [], 2);
            pmin_values = logP(ind);
            pmin_values = reshape(pmin_values, size(xx));
        end

        if exist('pmin_values')
            figure
            hold on
            mesh(xx, xy, exp(pmin_values));
            caxis([0, mean(mean(exp(pmin_values)))]);
        end
        
%         %plot over the representers used
%         f1 = figure;
%         hold on
%         f2 = figure;
%         hold on;
%         if se_dim == 0 && th_dim == 1
%             [~, ind] = sort(st_trials(:,end));
%             [xx xy] = ndgrid(st_trials(ind,end), zeros(params.Nb,1));
%             pmin_values = zeros(size(xx));
%             for i=1:size(xx,1)
%                 GPrel = problem.MapGP(GP, [plot_x(1:st_dim-1) xx(i,1)], params.LearnHypers);
%                 xy(i,:) = sort(zb_vec(:,end,1,ind(i)));
%                 pmin_values(i,:) = EstPmin(GPrel, [xy(i,:)'], 1000, randn(size(xy,2),1000));
%                 figure(f1);
%                 plot3(xx(i,:), xy(i,:), exp(pmin_values(i,:)));
%                 scatter3(xx(i,:), xy(i,:), exp(pmin_values(i,:)), '.');
%                 figure(f2);
%                 logw = lmb_vec(:,:,1,ind(i)) - lmb;
%                 plot3(xx(i,:), xy(i,:), exp(pmin_values(i,:)+logw'));   
%                 scatter3(xx(i,:), xy(i,:), exp(pmin_values(i,:)+logw'), '.');
%             end
%         else
% 
%             [~, ind] = sort(se_trials(:,end));
%             [xx xy] = ndgrid(se_trials(ind,end), zeros(params.Nb,1));
%             pmin_values = zeros(size(xx));
%             for i=1:size(xx,1)
%                 GPrel = problem.MapGP(GP, plot_x(sti), params.LearnHypers);
%                 xy(i,:) = zb_vec(:,end,ind(i),end);
%                 pmin_values(i,:) = EstPmin(GPrel, [xx(i,:)' xy(i,:)'], 1000, randn(size(xy,2),1000));
%                 figure(f1);
%                 plot3(xx(i,:), xy(i,:), exp(pmin_values(i,:)));
%                 scatter3(xx(i,:), xy(i,:), exp(pmin_values(i,:)), '.');
%                 %reweighted
%                 figure(f2);
%                 logw = lmb_vec(:,:,ind(i),end) - lmb;
%                 plot3(xx(i,:), xy(i,:), exp(pmin_values(i,:)+logw'));   
%                 scatter3(xx(i,:), xy(i,:), exp(pmin_values(i,:)+logw'), '.');
%             end
%             %%plot3(repmat(xx(i),1,in.Nb), pmin_values(i,:), zz');
%         end
    end
        
    %% eval function
    if ~params.output_off
        fprintf('evaluating function \n')
    end
    xp                = xatmin2';
    [yp,obsp]         = problem.sim_func([plot_x(sti) xp]);
    
    GP.x              = [GP.x ; xp ];
    if (params.Normalize)
        GP.y = GP.y + GP.offset;
        GP.y              = [GP.y ; yp ];
        GP.offset = mean(GP.y, 1);
        GP.y = GP.y - GP.offset;
    else
        GP.y              = [GP.y ; yp ];
    end
    GP.obs           = [GP.obs; obsp];
    %GP.dy             = [GP.dy; dyp];
    GP.K              = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
    GP.cK             = robustchol(GP.K);
    
    % helpers for later plots: include st for GP samples (only se and th)
    GP_full_x = repmat(plot_x, size(GP.x,1), 1);
    GP_full_x(:,[sei thi]) = GP.x;
   
    %% optimize hyperparameters
    % THIS is not necessary here. MapGP will always optimize hyp if needed
     if params.LearnHypers
        GP = problem.MapGP(GP, [], params.LearnHypers);         
%         minimizeopts.length    = 10;
%         minimizeopts.verbosity = 1;
%         GP.hyp = minimize(GP.hyp_initial,@(x)params.HyperPrior(x,GP.x,GP.y),minimizeopts);
%         GP.K   = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
%         GP.cK  = chol(GP.K);
        if ~params.output_off

         fprintf 'hyperparameters optimized.'
         display(['length scales: ', num2str(exp(GP.hyp.cov(1:end-1)'))]);
         display([' signal stdev: ', num2str(exp(GP.hyp.cov(end)))]);
         display([' noise stddev: ', num2str(exp(GP.hyp.lik))]);
        end     
     end
    
    %% evaluate over contexts
    s_vec = zeros(prod(params.Neval([sti sei])),st_dim+se_dim);
    theta_vec = zeros(prod(params.Neval([sti sei])),th_dim);
    val_vec = zeros(size(theta_vec,1),1);
    pred_vec = zeros(size(theta_vec,1),1);
    
    if ~mod(numiter, params.EvalModulo)
        if ~params.output_off
            fprintf('evaluate over contexts\n')
        end

        eval_st_vect = zeros(prod(params.Neval(sti)),st_dim);
        eval_se_vect = zeros(prod(params.Neval(sei)),se_dim);

        %eval_st_vect = evalvectfun(params.xmin(sti), params.xmax(sti), params.Neval(sti));
        s_cell = evalgridfun(params.xmin(sti), params.xmax(sti), params.Neval(sti));
        for i=1:st_dim
            eval_st_vect(:,i) = s_cell{i}(:)';
        end
 
        %eval_se_vect = evalvectfun(params.xmin(sei), params.xmax(sei), params.Neval(sei));
        s_cell = evalgridfun(params.xmin(sei), params.xmax(sei), params.Neval(sei));
        for i=1:se_dim
            eval_se_vect(:,i) = s_cell{i}(:)';
        end
        %s_cell = evalgridfun(xmin([sti sei]), xmax([sti sei]), params.Neval([sti sei]));

        k=1;
        for i=1:size(eval_st_vect,1)
            GPrel = problem.MapGP(GP, eval_st_vect(i,:), params.LearnHypers);
            for j=1:size(eval_se_vect,1)
                acqfun = @(theta)(gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y, [eval_se_vect(j,:) theta']));
                [theta, val] = ACESpolicy(acqfun, [params.xmin(thi)' params.xmax(thi)']);
                theta_vec(k,:) = theta;
                pred_vec(k,:) = val;
                s_vec(k,:) = [eval_st_vect(i,:) eval_se_vect(j,:)];
                k=k+1;
            end
        end
        if (~params.output_off && ~mod(numiter, PlotModulo.real))
            problem.PrintOn=true;
        end
        val_vec = problem.sim_eval_func([s_vec theta_vec]);
        problem.PrintOn=false;

        
        linstat.evaluated(numiter,:) = 1;
        if ~params.output_off
            current_performance = mean(val_vec)
        end
    else
        linstat.evaluated(numiter,:) = 0;
    end
    
    %linstat.theta_s(numiter, :) = theta_vec(:)';
    linstat.R_s(numiter, :) = val_vec(:)';
    linstat.R_mean(numiter, :) = mean(val_vec);
    linstat.GP(numiter,:) = GP.hyp;
    
    %% plot current theta policy
    % plot on real values
    if PlotModulo.policy && ~mod(numiter, PlotModulo.policy)
        plotgrid = evalgridfun(params.xmin(end-1:end), params.xmax(end-1:end), [25 25]);
        real_val = arrayfun(@(varargin)(problem.sim_plot_func([plot_x(1:end-2) varargin{:}])), plotgrid{:});
        
        figure
        mesh(plotgrid{1}, plotgrid{2}, real_val);
        hold on;
        
        scatter3(GP_full_x(:,end-1), GP_full_x(:,end), problem.sim_plot_func(GP_full_x), 'ro');
        if (th_dim == 1)
            if(s_dim == 1)
                tempi = 1:size(s_vec,1);
            else
                tempi = s_vec(:,1:s_dim-1)==plot_x(1:s_dim-1);
            end
            scatter3(s_vec(tempi,end), theta_vec(tempi,1), val_vec(tempi,:), 'y*');
        end
    end
    
    %plot on current GP belief
    if PlotModulo.policy && ~mod(numiter, PlotModulo.policy)
        plotgrid = evalgridfun(params.xmin(end-1:end), params.xmax(end-1:end), [100 100]);
        
        if th_dim+se_dim < 2
            full_m = zeros(100,100);
            full_s2 = zeros(100,100);
            for i=1:size(plotgrid{1},1)
                GPrel = problem.MapGP(GP, [plot_x(1:st_dim-1) plotgrid{1}(i,1)], params.LearnHypers);
                [full_m(i,:), full_s2(i,:)] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y,...
                    [plotgrid{2}(i,:)']);
            end
            curr_m = zeros(size(s_vec,1),1);
            curr_s2 = zeros(size(s_vec,1),1);
            for i=1:size(s_vec,1)
                GPrel = problem.MapGP(GP, s_vec(i,sti), params.LearnHypers);
                [curr_m(i,:) curr_s2(i,:)] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y, [s_vec(i,sei) theta_vec(i,:)]);
            end
        else
            GPrel = problem.MapGP(GP, plot_x(sti), params.LearnHypers);
            
            [full_m full_s2] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y, ...
                [repmat(plot_x(st_dim+1:end-2),numel(plotgrid{1}),1) plotgrid{1}(:) plotgrid{2}(:)]);
            full_m = reshape(full_m, size(plotgrid{1}));
            full_s2 = reshape(full_s2, size(plotgrid{1}));
            
            [curr_m curr_s2] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y, ...
                [s_vec(:,sei) theta_vec]);
        end
        
        figure
        mesh(plotgrid{1}, plotgrid{2}, full_m);
        hold on;
        scatter3(GPrel.x(:,end-1), GPrel.x(:,end), GPrel.y, 'ro');

        %scatter3(GP_full_x(:,end-1), GP_full_x(:,end), problem.sim_plot_func(GP_full_x), 'ro');
        full_vec = [s_vec theta_vec];
        scatter3(full_vec(:,end-1), full_vec(:,end), curr_m, 'y*');
        % %scatter3(out.FunEst(numiter,1), out.FunEst(numiter,2), in.f(out.FunEst(numiter,:)), 'r*');
        
        % compera with matlab's GP method
        % result is different (worse initially), i couldnt figure out why
        % PROBABLY default mean function is mean(y) rather than 0
%         gprMdl = fitrgp(GPrel.x, GPrel.y, ...
%                 'Basis','none','FitMethod','exact',...
%                 'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
%                 'KernelParameters',[params.sigmaM0; params.sigmaF0].^2,...
%                 'Sigma',params.sigma0, ...
%                 ...%'SigmaLowerBound', 1e-1*std(GPrel.y), ...
%                 'Standardize',0);
%         [full_m full_s2] = gprMdl.predict([repmat(plot_x(st_dim+1:end-2),numel(plotgrid{1}),1) plotgrid{1}(:) plotgrid{2}(:)]);    
%           full_m = reshape(full_m, size(plotgrid{1}));
%             full_s2 = reshape(full_s2, size(plotgrid{1}));
%         figure
%         mesh(plotgrid{1}, plotgrid{2}, full_m);
%         hold on;
%         scatter3(GP_full_x(:,end-1), GP_full_x(:,end), problem.sim_plot_func(GP_full_x), 'ro');
            
     end
    drawnow;
    
    %% FOR DEBUG
    %print a specific GP mapping for ST
    if (false)
        i=90;
        GPrel = problem.MapGP(GP, eval_st_vect(i,:), params.LearnHypers);
        figure
        hold on
        scatter(GPrel.x, GPrel.y);
        [m s2] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y,...
            [plotgrid{2}(i,:)']);
        plot(plotgrid{2}(i,:)', m);
        plot(plotgrid{2}(i,:)', m+2*s2, 'r-');
        plot(plotgrid{2}(i,:)', m-2*s2, 'r-');
        scatter(theta_vec(i,end), val_vec(i,end), 'r*');
        scatter(theta_vec(i,end), pred_vec(i,end), 'y*');
    end
    
    %print a specific GP mapping for SE
    if (false)
        i=90;
        GPrel = GP;
        %GPrel.x = [GPrel.x; plotgrid{1}(i,1), 0.5];
        %GPrel.y = [GPrel.y; 0];
        figure
        hold on
        scatter3(GPrel.x(:,1), GPrel.x(:,2), GPrel.y);
        [m s2] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y,...
            [plotgrid{1}(i,:)' plotgrid{2}(i,:)']);
        plot3(plotgrid{1}(i,:)', plotgrid{2}(i,:)', m);
        plot3(plotgrid{1}(i,:)', plotgrid{2}(i,:)', m+2*s2, 'r-');
        plot3(plotgrid{1}(i,:)', plotgrid{2}(i,:)', m-2*s2, 'r-');
        %scatter(theta_vec(i,end), val_vec(i,end), 'r*');
        %scatter(theta_vec(i,end), pred_vec(i,end), 'y*');
    end
    
    converged = params.ConvergedFunc();
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
linstat.st = [];
linstat.se = GP.x(:,sei-st_dim);
linstat.theta = GP.x(:,thi-st_dim);
linstat.outcome = GP.obs;

stats.last_R_mean = linstat.R_mean(end);
stats.lastGP = GP;

if params.ReturnOptimal
    [r_opt, r_worst] = problem.get_optimal_r(params.Neval);
    while max(size(r_opt)) > 1
        r_opt = mean(r_opt);
        r_worst = mean(r_worst);
    end
    stats.R_opt = r_opt;
    stats.R_worst = r_worst;
    linstat.R_mean = (linstat.R_mean - r_worst)/(r_opt-r_worst);
    
end

if ~params.output_off
    figure
    plot(linstat.R_mean);
end

end


