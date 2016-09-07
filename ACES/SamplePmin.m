function [sample, ind] = EstPmin(GP, zb)

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

[Mb,~]    = GP_moments(GP,zb);

mm = Mb + cov_multiplier*randn(size(zb,1),1);
[~, minind] = min(mm);
ind = minind;
sample = zb(ind, :);

end