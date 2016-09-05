function [ GPnew ] = MapFunc( GP, st, r_func )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

GPnew = GP;

if size(st)
    for i=1:size(GP.x,1)
        GPnew.y(i,:) = r_func(st, GP.x(i,:), GP.obs(i,:));
    end
    
    GPnew.K              = k_matrix(GPnew,GPnew.x) + diag(GP_noise_var(GPnew,GPnew.y));
    GPnew.cK             = chol(GPnew.K);

end


end

