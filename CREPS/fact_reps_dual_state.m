function [g, dg] = fact_reps_dual_state(etatheta, epsilon, R, Phi, w)

eta = etatheta(1);
theta = etatheta(2:end);

N = length(R);
V = Phi*theta(:);
MR = max(R-V);
Zhat = 1/N* w.* exp((R-V - MR)/eta);

g = eta*log(sum(Zhat)) + MR + eta*epsilon + mean(Phi, 1)*theta(:);
dgeta = epsilon + log(sum(Zhat)) + MR/eta - sum(Zhat.*(R-V))/eta/sum(Zhat);
dgtheta = mean(Phi, 1)' - Phi'*Zhat/sum(Zhat);

if isnan(g)
    g = 1e16;
end
if g > 1000000
    g = 1000000 + log(g-1000000+1);
end

dg = [dgeta, dgtheta'];


%if any(isnan([g, dg]))
%	keyboard
%end

