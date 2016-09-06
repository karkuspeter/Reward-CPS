function [ dHp ] = LossFunction( GP, logP, zb, lmb, x, params)
%U returns relative entropy




lPred = PredPmin(GP, zb, x, params);

% this is not ACES!

H   = - sum(exp(logP) .* (logP + lmb));           % current Entropy 
newH = -sum(exp(lPred) .* bsxfun(@plus,lPred,lmb),1);

if(H>0 || any(newH > 0))
    disp('Negative KL value');
end
%mlPred = mean(lPred,2);
%dHp = -sum(exp(mlPred) .* bsxfun(@plus,mlPred,lmb),1) - H; % @minus? If you change it, change it above in H, too!
%

dHp_full = newH - H; % @minus? If you change it, change it above in H, too!
dHp = mean(dHp_full,2);

%dHp = mean(dHp_all);
%dHp = mean(lPred, 2);

end

