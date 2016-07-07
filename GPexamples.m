
%Implements BO-CPS
%   Detailed explanation goes here

kappa = 0.5;

% %% example with dataset
% tbl = readtable('abalone.data','Filetype','text','ReadVariableNames',false);tbl.Properties.VariableNames = {'Sex','Length','Diameter','Height','WWeight','SWeight','VWeight','ShWeight','NoShellRings'};
% tbl(1:7,:)
% 
% gprMdl = fitrgp(tbl,'NoShellRings','KernelFunction','ardsquaredexponential',...
%       'FitMethod','sr','PredictMethod','fic','Standardize',1)
% ypred = resubPredict(gprMdl);
% 
% figure();
% plot(tbl.NoShellRings,'r.');
% hold on
% plot(ypred,'b');
% xlabel('x');
% ylabel('y');
% legend({'data','predictions'},'Location','Best');
% axis([0 4300 0 30]);
% hold off;
% 
% L = resubLoss(gprMdl)

%% simple example, 1D

rng(0,'twister'); % For reproducibility
n = 1000;
bounds = [-10, 10];
x = linspace(bounds(:,1),bounds(:,2),n)';
y = 1 + x*5e-2 + sin(x)./x + 0.2*randn(n,1);

gprMdl = fitrgp(x,y,'Basis','linear',...
      'FitMethod','exact','PredictMethod','exact');
  
[ypred, ystd] = resubPredict(gprMdl);
  
plot(x,y,'b.');
hold on;
%plot(x,ypred,'r','LineWidth',1.5);
plot_confidence(x, ypred, ystd);
xlabel('x');
ylabel('y');
legend('Data','GPR predictions');
hold off

%% find next theta
f = @(x)(acq_func(gprMdl, x, kappa));
[minval1,xatmin1,hist] = Direct(struct('f', f), bounds);

% refine by BFGS
[xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton'));

figure(2)
plot(x, acq_func(gprMdl, x, kappa));
hold on, plot(xatmin1, minval1, 'ko')
hold on, plot(xatmin2, minval2, 'ro')



%% 2D simple example

rng(0,'twister'); % For reproducibility
n = 40;
bounds = [-10, 10; ...
          -10, 10];
[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), n), ...
                    linspace(bounds(2,1),bounds(2,2), n));
y = 1 + x1*5e-2 + x2*5e-2 + sin(x1)./x1 + sin(x2)./x2 + 0.2*randn(n,n);
y = y.*10;

Xtrain = reshape(cat(2, x1, x2), [], 2);
ytrain = reshape(y, [], 1);
D = cat(2, Xtrain, ytrain);

sigma0 = 1;
sigmaF0 = sigma0;
d = size(Xtrain,2);
sigmaM0 = 0.2*ones(d,1);
gprMdl = fitrgp(Xtrain,ytrain,'Basis','constant','FitMethod','exact',...
'PredictMethod','exact','KernelFunction','ardsquaredexponential');%,...
%'KernelParameters',[sigmaM0;sigmaF0], 'Sigma',sigma0,'Standardize',1);

% show prediction
[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
                    linspace(bounds(2,1),bounds(2,2), 100));
Xplot = reshape(cat(2, x1, x2), [], 2);
[ypred, ystd] = predict(gprMdl, Xplot);
Yplot = reshape(ypred, size(x1,1), []);

figure;
mesh(x1(1,:)', x2(:,1), Yplot);
hold on
scatter3(D(:,1), D(:,2), D(:,3));
xlabel('x');
ylabel('y');
legend('Data','GPR predictions');
hold off

%% find next theta
f = @(x)(acq_func(gprMdl, x', kappa));
[minval1,xatmin1,hist] = Direct(struct('f', f), bounds);

% refine by BFGS
[xatmin2, minval2] = fminunc(f, xatmin1, optimoptions('fminunc','Algorithm','quasi-newton'));