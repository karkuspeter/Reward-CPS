function val = ACES2(GP, zb, lmb, x, st_dim, mapfunc, params, seed)

if nargin < 8
    use_seed = 0;
    seed = 0;
else
    use_seed = 1;
end

if use_seed
    rng(seed);  %if every time called with same seed ACES will be the same
end

s = x(1);
%offset = [-0.2; -0.1; 0; 0.1; 0.2];
offset = 0;
w = normpdf(offset, 0, exp(GP.hyp.cov(1)/2)) .* [0.5; 0.5; 1; 0.5; 0.5];
s_vec = offset + s;
w = w(s_vec >= params.xmin(1) & s_vec <= params.xmax(1));
s_vec = s_vec(s_vec >= params.xmin(1) & s_vec <= params.xmax(1));
val_vec = zeros(size(s_vec,1),1);

for i=1:length(s_vec)
    
    GPrel = mapfunc(GP, s_vec(i,1:st_dim));
    logP = EstPmin(GPrel, zb, params.S, randn(size(zb,1), params.S));
    val_vec(i) = LossFunction(GPrel, logP, zb, lmb, x(st_dim+1:end), params);
end

val = sum(w.*val_vec)/sum(w);

end