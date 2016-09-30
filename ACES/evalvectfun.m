function evalvect = evalvectfun(xmin, xmax, Neval)
if isempty(xmin)
    evalvect = zeros(1,0);
    return
end
evalgrid = cell(1, length(xmin));
gridvect = cell(1, length(xmin));
for iiii=1:length(xmin)
    gridvect{iiii} = linspace(xmin(iiii), xmax(iiii), Neval(iiii))';
end
[evalgrid{:}] = ndgrid(gridvect{:});

evalvect = [evalgrid{:}(:)];
end