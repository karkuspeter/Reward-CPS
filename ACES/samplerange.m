function [ samples ] = samplerange( xmin, xmax, N )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here`
D = length(xmin);
if ~D
    samples = zeros(N, 0);
else
    samples = repmat(xmin, N, 1) + rand(N,D).*repmat((xmax - xmin), N, 1);
end

end

