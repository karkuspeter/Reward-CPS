function [samples, inds, w] = SamplePmin(GP, zb, Nb, replacement)

if nargin < 4
    replacement = false;
end
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
map = 1:size(zb,1);
inds = zeros(Nb, 1);
w = zeros(zb, 1);

i=1;
while i<=Nb
    mm = Mb + cov_multiplier*randn(size(Mb,1),1);
    [~, minind] = min(mm);
    
    % sample with replacement. w indicates the number i was sampled
    if replacement
        if w(minind) > 0
            % this has been sampled once, only increase w but not i
            w(minind) = w(minind) + 1;
        else
            w(minind) = 1; 
            inds(i,:) = minind;
            i = i+1;
        end
    else
        inds(i,:) = map(minind);
        
        % no replacement so have to remove sample
        Mb = Mb([1:minind-1 minind+1:end]);
        map = map([1:minind-1 minind+1:end]);
        cov_multiplier = cov_multiplier([1:minind-1 minind+1:end], [1:minind-1 minind+1:end] );

        i = i+1;
    end
        
end

samples = zb(inds, :);
w = w(inds,:);

end