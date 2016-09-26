function [ cx ] = robustchol( x )
%calls chol(x). If x is not pos def uses chol(nearestSPD(x))

[cx, p] = chol(x);
if p
    x = nearestSPD(x);
    cx = chol(x);
end

end

