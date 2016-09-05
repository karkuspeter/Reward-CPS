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
if ~isfield(in,'st_dim'); in.st_dim = 0; end;
if ~isfield(in,'se_dim'); in.se_dim = 1; end;
if ~isfield(in,'MapFunc'); in.MapFunct = {@(GP, s)(GP)}; end;

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
D = in.D;
st_dim = in.st_dim;
se_dim = in.se_dim;
s_dim = in.st_dim + in.se_dim;
th_dim = D-s_dim;
if s_dim ~= 1
    disp('Error: only 1d context supported');
    return;
end
sti = 1:st_dim;
sei = st_dim+1:st_dim+se_dim;
thi = s_dim+1:D;

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


params = struct('S', 1000, 'Ny', 10,  'Ntrial_st', 20, ...
    'Ntrial_se', 100, ...
    'Nn', 8, ...  % only the neares Nn out of Ntrial_se will be evaluated for a context se
    'Nb', in.Nb, ... %number of representers for p_min over theta space
    'Neval', [20, 100], ...
    'xmin', in.xmin, 'xmax', in.xmax, 'st_dim', in.st_dim, 'se_dim', in.se_dim);

plotswitch = struct('ACES', 0, 'pmin', 1, 'policy', 1);

if ~st_dim
    params.Ntrial_st = 1;
end
if ~se_dim
    params.Ntrial_se = 1;
end
    
% construct grid

evalgrid = evalgridfun(in.xmin, in.xmax, params.Neval);
if in.ReturnOptimal
    val_full = arrayfun(@(varargin)(in.f([varargin{:}])), evalgrid{:}); % this is a D dim array
    out.val_opt = val_full;
    for i_th=s_dim+1:D
        out.val_opt = min(out.val_opt, [], i_th);
    end
end

plot_x = (in.xmax + in.xmin)/2;

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
    st_trials = samplerange(in.xmin(sti), in.xmax(sti), params.Ntrial_st);
    se_trials = samplerange(in.xmin(sei), in.xmax(sei), params.Ntrial_se);
    %TODO: replace with sampling distribution that tries to cover range
    %rather than picking independently
    
        % sample belief and evaluation points
        %[zb,lmb]   = SampleBeliefLocations(GP,in.xmin,in.xmax,in.Nb,BestGuesses,in.PropFunc);
        %TODO should be Thompson sampling. This EI based is absolutely not
        %good
        
    zb = samplerange(in.xmin(thi), in.xmax(thi), params.Nb);
    
    lmb = -log(norm([in.xmin(thi)' in.xmax(thi)']))*ones(params.Nb,1);  %log of uniform measure, |I|^-1
    
    if th_dim == 1
       [zt,sorti] = sort(zb(:,end)); %theta line
         vals = [(zt(2)-zt(1))/2 + zt(1)-in.xmin(end);
                  (zt(3:end) - zt(1:end-2))/2;
                  in.xmax(end) - zt(end) + (zt(end)-zt(end-1))/2];
         vals(sorti) = vals; %restore order
         lmb = lmb + log(vals);
         %TODO this is not good, need better representers
    else
        disp('Note: for theta_dim > 1 this is not great\n');
    end
    
    % generate zb, logP vectors for each trial se
    zb_vec = zeros(in.Nb, se_dim+th_dim, params.Ntrial_se, params.Ntrial_st);
    lmb_vec = zeros(in.Nb, 1, params.Ntrial_se, params.Ntrial_st);
    logP_vec = zeros(in.Nb, 1, params.Ntrial_se, params.Ntrial_st);
    GP_cell = cell(params.Ntrial_st, 1);
    
    for i_st=1:params.Ntrial_st
        GP_cell{i_st} = in.MapFunc(GP, st_trials(i_st,:));
        for i_se=1:params.Ntrial_se
            zb_vec(:,:,i_se, i_st) = [repmat(se_trials(i_se,:),in.Nb,1) zb];
            lmb_vec(:,:,i_se, i_st) = lmb;
            logP_vec(:,:,i_se, i_st) = EstPmin(GP, zb_vec(:,:,i_se, i_st), params.S, randn(size(zb,1), params.S));  %joint_min(Mb_vec(:,:,i), Vb_vec(:,:,i), 1);          
        end
    end
       
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
 

        % construct ACES function for this GP
        rand_start = rng();
        % rand_start = rng('shuffle');
        %aces_f = @(x)(ACES(GP, logP_vec, zb_vec, lmb_vec, y_vec, x, trial_contexts, in.st_dim, params, rand_start));
        %aces_f = @(x)(ACES2(GP, zb, lmb, x, in.st_dim, in.MapFunc, params, rand_start));
        aces_f = @(x)(ACES3(GP_cell, logP_vec, zb_vec, lmb_vec, st_trials, se_trials, x, params, rand_start));
         
        
        [minval1,xatmin1,hist] = Direct(struct('f', @(x)(aces_f(x'))), [in.xmin([sei thi])' in.xmax([sei thi])'], struct('showits', 1, 'maxevals', 40));
        xrange = [in.xmax([sei thi])' - in.xmin([sei thi])']/10;
        [minval2,xatmin2,hist] = Direct(struct('f', @(x)(aces_f(x'))), [xatmin1-xrange xatmin1+xrange], struct('showits', 1, 'maxevals', 40));
        xatmin2
        
        %TODO only working until here with dynamic context size!!!
        
        %% print ACES function
        if plotswitch.ACES && se_dim+th_dim >= 2
            fprintf('plot ACES function\n')

            %for printing
            [xx, xy] = meshgrid(linspace(in.xmin(end-1),in.xmax(end-1),10)', linspace(in.xmin(end),in.xmax(end),10)');
            
            
            
            %indexhelper = ; %this will be empty matrix when only 1d theta+se space
            figure
            aces_values = arrayfun(@(a,b)(aces_f([plot_x(1:end-2) a b])), xx, xy);
            mesh(xx, xy, aces_values);
            
            hold on;
            scatter3(GP.x(:,end-1), GP.x(:,end), in.f(GP.x), 'ro');
            %scatter3(xstart(:,1), xstart(:,2), arrayfun(@(a,b)aces_f([a b]), xstart(:,1), xstart(:,2)), 'b*');
            scatter3(xatmin1(end-1), xatmin1(end), minval1, 'y*');
            scatter3(xatmin2(end-1), xatmin2(end), minval2, 'r*');
            drawnow;
        end

        %% plot pmin
        if plotswitch.pmin && (st_dim == 0 && se_dim == 1 && th_dim == 1)
            xx = linspace(in.xmin(1),in.xmax(1),100)';
            figure
            hold on
            for i_se=1:100
                zz = sort(zb(:,1));
                pmin_values(i_se,:) = EstPmin(GP, [repmat(xx(i_se),in.Nb,1) zz], 1000, randn(size(zb,1),1000));
                %%plot3(repmat(xx(i),1,in.Nb), pmin_values(i,:), zz');
            end

            [xx xy] = meshgrid(linspace(in.xmin(1),in.xmax(1),100)', zz);

            mesh(xx, xy, exp(pmin_values)');
            caxis([0, 0.5]);
        end
        
        % eval function
        fprintf('evaluating function \n')
        xp                = xatmin2';
        yp                = in.f(xp);
        
        GP.x              = [GP.x ; xp ];
        GP.y              = [GP.y ; yp ];
        %GP.dy             = [GP.dy; dyp];
        GP.K              = k_matrix(GP,GP.x) + diag(GP_noise_var(GP,GP.y));
        GP.cK             = chol(GP.K);
        
        % estimate minimum
        %MeanEsts(numiter,:) = sum(bsxfun(@times,zb,exp(logP)),1);
        %[~,MAPi]            = max(logP + lmb);
        %MAPEsts(numiter,:)  = zb(MAPi,:);
        
        %fprintf('finding current best guess\n')
        %[out.FunEst(numiter,:),FunVEst] = FindGlobalGPMinimum(BestGuesses,GP,in.xmin,in.xmax);
        % % is the new point very close to one of the best guesses?
        %[cv,ci] = min(sum(bsxfun(@minus,out.FunEst(numiter,:)./ell,bsxfun(@rdivide,BestGuesses,ell)).^2,2)./D);
        %if cv < 2.5e-1 % yes. Replace it with this improved guess
        %    BestGuesses(ci,:)  = out.FunEst(numiter,:);
        %else % no. Add it to the best guesses
        %    BestGuesses(size(BestGuesses,1)+1,:) = out.FunEst(numiter,:);
        %end
    
        
        %% evaluate over contexts
        fprintf('evaluate over contexts\n')
        plot_st_vect = evalvectfun(in.xmin(sti), in.xmax(sti), params.Neval(sti));
        plot_se_vect = evalvectfun(in.xmin(sei), in.xmax(sei), params.Neval(sei));
        %s_cell = evalgridfun(xmin([sti sei]), xmax([sti sei]), params.Neval([sti sei]));

        s_vec = zeros(prod(params.Neval([sti sei])),st_dim+se_dim);
        theta_vec = zeros(prod(params.Neval([sti sei])),th_dim);
        %val_vec = zeros(size(theta_vec,1),1);
        
        k=1;
        for i=1:size(plot_st_vect,1)
            %if st_dim
                GPrel = in.MapFunc(GP, plot_st_vect(i,:));
            %else
            %    GPrel = GP;
            %end
            for j=1:size(plot_se_vect,1)
                [theta, val] = ACESpolicy(GPrel, plot_se_vect(j,:), [in.xmin(thi)' in.xmax(thi)']);
                theta_vec(k,:) = theta;
                s_vec(k,:) = [plot_st_vect(i,:) plot_se_vect(j,:)];
                k=k+1;
            end
        end
        val_vec = in.f([s_vec theta_vec]);
        current_performance = sum(val_vec)
        out.val_vec(:,numiter) = val_vec;
        
        %% plot current theta policy
        if plotswitch.policy
            plotgrid = evalgridfun(in.xmin(end-1:end), in.xmax(end-1:end), [100 100]);
            real_val = arrayfun(@(varargin)(in.f([varargin{end-(th_dim==1):end}])), plotgrid{:});

            figure
            mesh(plotgrid{1}, plotgrid{2}, real_val);      
            hold on;
            GP_full_x = repmat(plot_x, size(GP.x,1), 1);
            GP_full_x(:,[sei thi]) = GP.x;
            scatter3(GP.x(:,end-1), GP.x(:,end), in.f(GP_full_x), 'ro');
            if (th_dim == 1)
                if(s_dim == 1)
                    tempi = 1:size(s_vec,1);
                else
                    tempi = s_vec(:,1:s_dim-1)==plot_x(1:s_dim-1);
                end
                scatter3(s_vec(tempi,end), theta_vec(tempi,1), val_vec(tempi,:), 'y*');
            end
            %scatter3(out.FunEst(numiter,1), out.FunEst(numiter,2), in.f(out.FunEst(numiter,:)), 'r*');    

            %figure
            %[my_m my_s2] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, [xx(:) xy(:)]);
            % mesh(xx,xy, reshape(my_m, size(xx)));
            %hold on;
            %scatter3(GP.x(:,1), GP.x(:,2), in.f(GP.x), 'ro');
            %[my_m my_s2] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, [s_vec theta_vec]);
            %scatter3(s_vec(:,1), theta_vec(:,1), my_m, 'y*');
            % %scatter3(out.FunEst(numiter,1), out.FunEst(numiter,2), in.f(out.FunEst(numiter,:)), 'r*');    
        end  
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

end

function evalgrid = evalgridfun(xmin, xmax, Neval)
    if isempty(xmin)
        evalgrid = cell(1,0);
        return
    end
    evalgrid = cell(1, length(xmin));
    gridvect = cell(1, length(xmin));
    for iiii=1:length(xmin)
        gridvect{iiii} = linspace(xmin(iiii), xmax(iiii), Neval(iiii))';
    end
    [evalgrid{:}] = ndgrid(gridvect{:});
end

function evalvect = evalvectfun(xmin, xmax, Neval)
    if isempty(xmin)
        evalvect = zeros(1,0);
        return
    end
    evalgrid = cell(1, length(xmin));
    gridvect = cell(1, length(xmin));
    for iiii=1:length(xmin)
        gridvect{iiii} = linspace(xmin(iiii), xmax(iiii), Neval(iiii))';
    end
    [evalgrid{:}] = ndgrid(gridvect{:});
    
    evalvect = [evalgrid{:}(:)];
end