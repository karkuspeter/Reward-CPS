function [h] = plot_confidence(x, y_mean, y_std, fillcolor, linecolor, func)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

colorOrder = get(gca, 'ColorOrder');

if nargin < 4
   fillcolor = colorOrder(mod(length(get(gca, 'Children')), size(colorOrder, 1))+1, :);

   % fillcolor = 'b'; %[7 7 7]/8;
end
if nargin < 5
   linecolor = colorOrder(mod(length(get(gca, 'Children')), size(colorOrder, 1))+1, :);
end
if nargin < 6
   func = @(x)(x);
end

hold on

f = [func(y_mean+2*y_std); flipdim(func(y_mean-2*y_std),1)];
h = fill([x; flipdim(x,1)], f, fillcolor);
set(h,'facealpha',.15)
set(h,'EdgeColor','None')
%set(h,'alphadata',.1)

h = plot(x, func(y_mean), 'Color', linecolor, 'LineWidth', 3)

end

