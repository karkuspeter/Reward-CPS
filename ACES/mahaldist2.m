function [ d2 ] = mahaldist2( x1, x2, Sinv )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    d2 =  (x1-x2) * Sinv * (x1-x2)';

end

