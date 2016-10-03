function [a, A, cov, rew, eta, theta, hist] = fact_reps_state(a, sigs, stFunc, seFunc, rewFunc, simFunc, epsilon, samples, episodes, etatheta)
% This script simulates the learning with contextual REPS
%
% Inputs:
% a: initial policy mean (Gaussian policy)
% sigs: the standard deviations of the Gaussian policy
% contextFunc(N): callable function, samples N context variables (e.g. contextFunc(5) gives 5 context variables)
% rewFunc(context, policy parameter): callable function, gives the reward
% epsilon: the relative entropy upper bound
% samples: for each policy update the algorithm samples "samples" amount of samples :)
% episodes: number of policy updates
% etatheta: initial Lagrangian parameters initialize with some small random
% number, eta must be positive!!!

options = optimset('Algorithm','active-set');
options = optimset(options, 'GradObj','on');
options = optimset(options, 'Display', 'off');
S = [];
r = [];
N = samples;
Nart = 100;
C = [];
hist = struct('a', [], 'A', [], 'cov', []);

dim = length(a);
cov = diag(sigs.^2);

dummyCe = seFunc(1);
dummyCt = stFunc(1);
se_dim = length(dummyCe);
st_dim = length(dummyCt);

A = zeros(dim, st_dim+se_dim);



for e = 1:episodes
    
    outcomes = [];
    rnew = zeros(N*Nart, 1);
    w = zeros(N*Nart, 1);
    
    for i = 1:N
        C(i, :) = [stFunc(1)', seFunc(1)'];
        Snew(i, :) = mvnrnd(a + A*C(i, :)', cov, 1)';
        
        %simulate outcome
        [~, outcomes(i,:)] = simFunc(C(i, :), Snew(i, :));
        
        %compute reward
        rnew(i) = rewFunc(C(i, :), Snew(i, :), outcomes(i,:));
        w(i) = 1;
    end
    
    %for artificial samples
    for i = 2:Nart
        for j=1:N
            idx = (i-1)*N + j;
            C(idx, :) = [stFunc(1)', C(j, st_dim+1:end)];
            Snew(idx, :) = mvnrnd(a + A*C(idx, :)', cov, 1)';

            rnew(idx) = rewFunc(C(idx, :), Snew(j, :), outcomes(j,:));
            
            prob_sample = mvnpdf(Snew(j,:)', a + A*C(j, :)', cov);
            prob_current = mvnpdf(Snew(j,:)', a + A*C(idx, :)', cov);
            w(idx) = prob_sample/prob_current;
        end
    end
    
    % you can reuse old samples here
% 	S = [S; Snew];
% 	r = [r; rnew];
    
	
	S = Snew;
	r = rnew;
    
    ixBad = find(isnan(r));
    ixGood = setdiff(1:length(r), ixBad);
    
    if ~isempty(ixBad)
        warning(['There are ', num2str(length(ixBad)), ' NaN values in the reward! These samples will be discarded'])
    end
    
    C = C(ixGood, :);
    S = S(ixGood, :);
    r = r(ixGood);
    w = w(ixGood);
    
    % using linear + quadratic features
    Phi = [ones(N*Nart, 1), C, C.^2];

	% With Gradient
	objfun = @(etatheta) reps_dual_state(etatheta, epsilon, r, Phi);
    
    try % There might be some numerical problems
        etatheta_prev = etatheta;
        etatheta = fmincon(objfun, etatheta(:), diag([-1; ones(3, 1)]), [-.0001; Inf; Inf; Inf], [], [], [], [], [], options);
        if isnan(etatheta)
            disp('rewards')
            r
            error('NaN eta')
        end
    catch err
        disp(err.identifier);
        disp(err.message);
        if mean(r)/std(r) > 100
            disp('Optimal solution found!')
            break
        end

    end

    
    eta = etatheta(1);
    theta = etatheta(2:end);
    
    V = Phi*theta;
        
	p = exp((r - V)/eta).*w;
    p = p/sum(p);
    
    % Finding the ML solution for a linear (in context) Gaussian policy
    CC = [ones(size(C,1), 1), C];
    aA = (CC' * diag(p) *CC)\CC'*diag(p)*S;
     
    a = aA(1, :)';   
    A = aA(2:end, :)';

    bsxfun(@minus, S, a');
    dummy = S - bsxfun(@plus, a , A*C')';
    cov = bsxfun(@times, p, dummy)'*dummy / sum(p);

    
    if any(isnan(a))
        
        
        a = aprev;
        A = Aprev;
        eta = etaprev;
        theta = thetaprev; 
        break
    end
    
    aprev = a;
    Aprev = A;
    etaprev = eta;
    thetaprev = theta;

    rew(e) = mean(r);
    hist.a(e,:) = a(:)';
    hist.A(e,:) = A(:)';
    hist.cov(e,:) = cov(:)';
    
% 	rexp = r(:)'*p;
% 	rvar = (r(:)-rexp)'*((r(:)-rexp).*p);
% 
% 	savem = [savem; mn', diag(cn)'.^.5, cn(2, 1)/prod(diag(cn).^.5), eta, rexp, Dkl, iDkl, rvar];
%     disp(['Episode #', num2str(e), ', mean reward: ', num2str(rewFunc(a(:)'))]);
end
