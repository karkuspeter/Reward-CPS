function val = ACES(GP, logP_vec, zb_vec, lmb_vec, y_vec, x, s_vec, st_dim, params, seed)
% s_vec: trial contexts N * s_dim
% logP vec: logp(x=xmin) zb_count * 1 * N

if nargin < 8
    use_seed = 0;
    seed = 0;
else
    use_seed = 1;
end

% xnew - predict pmin given a new sample at xnew
sdim = size(s_vec,2);
s = x(1:sdim);

% if sdim > 1
%     disp('currently only for 1D');
%     return;
% end

dm = zeros(size(s_vec,1),1);
for i=1:size(s_vec,1)
    dm(i) = mahaldist2(s_vec(i,:), s, GP.invL(sdim+1:end, sdim+1:end));
end

[sortedX,sortingIndices] = sort(dm,'ascend');

% figure
% scatter(trial_contexts(:,1), trial_contexts(:,2), 'bo');
% hold on
% scatter(contexts(:,1), contexts(:,2), 'r*');
% scatter(s(1), s(2));

val = zeros(params.Nn, 1);
parfor i=1:params.Nn    %parfor on lab PC got double speed with 4 cores, Nn = 4
    if use_seed
        rng(seed);  %if every time called with same seed ACES will be the same
    end
    s_dex = sortingIndices(i); %this should work for higher dim context too
    %context = s_vec(s_dex, :);
    logP = logP_vec(:, :, s_dex);
    zb = zb_vec(:, :,s_dex);
    lmb = lmb_vec(:,:, s_dex);
    xrel = x(st_dim+1:end);
    GPrel = GP;
    if st_dim
        GPrel.y = y_vec(:,:,s_dex);
    end
    
    val(i) = LossFunction(GPrel, logP, zb, lmb, xrel, params);
    
    %TODO lmb is for the full context+theta space,
    % should be for p(xmin | context)
end

val = sum(val);

end