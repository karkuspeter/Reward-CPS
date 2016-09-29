function val = acq_func_bo3(GP_cell, offset_vec, st_trials, se_trials, x, params)

st_dim = size(st_trials, 2);
se_dim = size(se_trials, 2);

x_se = x(1 : se_dim);
x_th = x(se_dim + 1 : end);

% get nearest Nn se context from se_trials
if se_dim
    dm = zeros(size(se_trials,1),1);
    for i=1:size(se_trials,1)
        dm(i) = mahaldist2(se_trials(i,:), x_se, GP_cell{1}.invL(st_dim+1:st_dim+se_dim, st_dim+1:st_dim+se_dim));
        %Note: GP_cel{1} assumes that invL will be the same for all st GP-s.
    end
    [sortedX, sortedIndices] = sort(dm,'ascend');
    rel_se_inds = sortedIndices(1:params.Nn);
else
    rel_se_inds = 1;
end

%linearize indexes to execute for st and se, so we can use parfor
[a,b] = ndgrid(1:size(st_trials,1), 1:size(rel_se_inds,1));
eval_inds = [a(:) b(:)];

val = zeros(size(eval_inds,1), 1);
for i=1:size(eval_inds,1)    %parfor on lab PC got double speed with 4 cores, Nn = 4
    i_st = eval_inds(i,1);
    i_se = rel_se_inds(eval_inds(i,2));
    %just for debugging
    %st = st_trials(i_st,:)
    %se = se_trials(rel_se_inds(eval_inds(:,2)),:)
    GP = GP_cell{i_st};
    val(i) = acq_func_bo2(GP, [x_se x_th], params.kappa) - offset_vec(1,1,i_se, i_st);
end

val = mean(val);

end