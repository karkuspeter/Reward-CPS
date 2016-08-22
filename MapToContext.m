function [D, Rstar] = MapToContext(Dfull, context, r_func, isPopulate, Npopulate, st_bounds)
%D = zeros(datalength, se_dim, theta_dim);
%Rstar = zeros(datalength,1);
D = [];
Rstar = [];
temp_seed = rng();
for i=1:size(Dfull.theta,1)
    if size(Dfull.se)
        sfull = [context, Dfull.se(i,:)];
        se = Dfull.se(i,:);
    else
        se = [];
        sfull = context;
    end
    newD = [context, se, Dfull.theta(i,:)];
    newR = zeros(1+Npopulate,1);
    newR(1) = r_func([context, se], Dfull.theta(i,:), Dfull.outcome(i,:));
    for j=1:Npopulate
        art_context = ((st_bounds(:,2)-st_bounds(:,1)).*rand(size(st_bounds,1),1) + st_bounds(:,1))';
        % smarter way: gaussian around the outcome so samples are more
        % relevant
        newR(j+1) = r_func([art_context, se], Dfull.theta(i,:), Dfull.outcome(i,:));
        newD(j+1, :) = [art_context, se, Dfull.theta(i,:)];
    end
    Rstar = [Rstar; newR]; 
    D = [D; newD];
end
if ~isPopulate
    D = D(:, size(context,2)+1:end);
end
rng(temp_seed);
end