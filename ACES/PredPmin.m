function pmin_pred = PredPmin(GP, zb, xnew, params)

% xnew - predict pmin given a new sample at xnew

%Ny = 200;
pmin_pred = zeros(size(zb,1),params.Ny);
%[m my_s2] = gp(GP.hyp, [], [], GP.covfunc, GP.likfunc, GP.x, GP.y, xnew);
[m, V] = GP_moments(GP, xnew);
cholV = robustchol(V);

for i=1:params.Ny
    ynew = m + randn(size(m))*cholV; % sample from bivariate norm distr
    GPnew = GP;
    GPnew.x = [GPnew.x; xnew];
    GPnew.y = [GPnew.y; ynew];        
    GPnew.K              = k_matrix(GPnew,GPnew.x) + diag(GP_noise_var(GPnew,GPnew.y));
    GPnew.cK             = robustchol(GPnew.K);
    
    pmin_pred(:,i) = EstPmin(GPnew, zb, params.S, randn(size(zb,1),params.S));
    %TODO random numbers should be generated once to decrease noise
end


end