function logP = EstPmin(GP, zb, S, randm)

kTT = feval(GP.covfunc{:}, GP.hyp.cov, zb);
kTI = feval(GP.covfunc{:}, GP.hyp.cov, zb, GP.x);
kII = feval(GP.covfunc{:}, GP.hyp.cov, GP.x);
sigma = exp(2*GP.hyp.lik)*eye(size(GP.x,1));

% Posterior covariance matrix
kPost = kTT - kTI/(kII+sigma)*kTI';
% if not positive semi-definite then find the nearest one
%TODO how does this compromise the matrix... does this make sense?
[cov_multiplier, p] = chol(kPost);
if p
    kPost = nearestSPD(kPost);
    cov_multiplier = chol(kPost);
end
cov_multiplier = cov_multiplier';

P = zeros(size(zb,1),1);
[Mb,~]    = GP_moments(GP,zb);

for i=1:S
    mm = Mb + cov_multiplier*randm(:,i);
    %TODO should have rand for each dimension - NO, THESE ARE VALUES
    [~, minind] = min(mm);
    P(minind) = P(minind) + 1;
end

logP = log(P);
logP(isinf(logP)) = -50;
logP  = logP - logsumexp(logP);  %normalize
end