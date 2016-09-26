function logP = EstPmin(GP, zb, S, randm)




kTT = feval(GP.covfunc{:}, GP.hyp.cov, zb);
kTI = feval(GP.covfunc{:}, GP.hyp.cov, zb, GP.x);
%kII = feval(GP.covfunc{:}, GP.hyp.cov, GP.x);
%sigma = exp(2*GP.hyp.lik)*eye(size(GP.x,1));

cKI = inv(GP.cK);
% Posterior covariance matrix
%kPost = kTT - kTI/(kII+sigma)*kTI';
kPost = kTT - kTI*(cKI*cKI')*kTI';
% if not positive semi-definite then find the nearest one
%TODO how does this compromise the matrix... does this make sense?
cov_multiplier = robustchol(kPost);

cov_multiplier = cov_multiplier';

P = zeros(size(zb,1),1);
[Mb,~]    = GP_moments(GP,zb);
%figure; hold on;
% for i=1:S
%     mm = Mb + cov_multiplier*randm(:,i);
%     %TODO should have rand for each dimension - NO, THESE ARE VALUES
%     [~, minind] = min(mm);
%     P(minind) = P(minind) + 1;
%  %   if ~mod(i,100)
%  %       plot(zb(:,end), mm); 
%  %       scatter(zb(:,end), mm, '.');
%  %   end
% end



mm = repmat(Mb, 1, S) + cov_multiplier*randm;
[~, minind] = min(mm, [], 1);
for i=1:S
    P(minind(i)) = P(minind(i)) + 1;
end

%figure; hold on;
%plot(zb(:,end), mm(:,1:25))

logP = log(P);
logP(isinf(logP)) = -50;
logP  = logP - logsumexp(logP);  %normalize
end


% %% tried without nearest SPD. It gives same shapes but very large variance
% P2 = zeros(size(zb,1),1);
% mm2 = repmat(Mb, 1, S) + (kPost')*randm;
% [~, minind2] = min(mm2, [], 1);
% for i=1:S
%     P2(minind2(i)) = P2(minind2(i)) + 1;
% end
% 
% [Mb Vb] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, zb);
% 
% figure; hold on;
% plot(zb(:,end), mm2(:,1:25))
% plot(zb(:,end), Mb, 'LineWidth',2)
% plot(zb(:,end), Mb+2*Vb, '--' ,'LineWidth',2)