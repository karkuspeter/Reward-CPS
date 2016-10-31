
clear all, close all
write_fig = 0;
disp(' ')

meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 1];
covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
disp(' ')

disp('n = 20;')
n = 20;
disp('x = gpml_randn(0.3, n, 1);')
x = gpml_randn(0.3, n, 1);
disp('K = feval(covfunc{:}, hyp.cov, x);')
K = feval(covfunc{:}, hyp.cov, x);
disp('mu = feval(meanfunc{:}, hyp.mean, x);')
mu = feval(meanfunc{:}, hyp.mean, x);
disp('y = chol(K)''*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);')
y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);
 
figure(1)
set(gca, 'FontSize', 24)
disp(' '); disp('plot(x, y, ''+'')')
plot(x, y, '+', 'MarkerSize', 12)
axis([-1.9 1.9 -0.9 3.9])
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f1.eps; end

%% sect 2
disp(' ')
disp('nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)')
nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)
disp(' ')

disp('z = linspace(-1.9, 1.9, 101)'';')
z = linspace(-1.9, 1.9, 101)';
disp('[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);')
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);


kTT = feval(covfunc{:}, hyp.cov, z);
kTI = feval(covfunc{:}, hyp.cov, z,x);
kII = feval(covfunc{:}, hyp.cov, x);
sigma = exp(2*hyp.lik)*eye(max(size(x))); 

% Posterior covariance matrix
kPost = kTT - kTI/(kII+sigma)*kTI';

mm = m + (chol(kPost)')*randn(101,1);



figure(2)
set(gca, 'FontSize', 24)
disp(' ')
disp('f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];') 
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8);')
fill([z; flipdim(z,1)], f, [7 7 7]/8);

disp('hold on; plot(z, m); plot(x, y, ''+'')')
hold on; plot(z, m, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
for i=1:10
    mm = m + (chol(kPost)')*randn(101,1);
    plot(z, mm, 'LineWidth', 1);
end
axis([-1.9 1.9 -0.9 3.9])
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f2.eps; end
disp(' '); 

%% sample a 1D function from a 2D GP
meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [0.5; 0.5; 1];
covfunc = {@covSEard}; ell = 1/4; sf = 1; hyp.cov = log([ell; ell; sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

x = gpml_randn(0.3, n, 2);
K = feval(covfunc{:}, hyp.cov, x);
mu = feval(meanfunc{:}, hyp.mean, x);
y = chol(K)'*gpml_randn(0.15, n, 1) + mu + exp(hyp.lik)*gpml_randn(0.2, n, 1);

%1.4395
z = [1.2*ones(101,1), linspace(-1.9, 1.9, 101)'];
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

kTT = feval(covfunc{:}, hyp.cov, z);
kTI = feval(covfunc{:}, hyp.cov, z,x);
kII = feval(covfunc{:}, hyp.cov, x);
sigma = exp(2*hyp.lik)*eye(max(size(x))); 
% Posterior covariance matrix
kPost = kTT - kTI/(kII+sigma)*kTI';

figure
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8);')
fill([z; flipdim(z,1)], f, [7 7 7]/8);
hold on; plot(z, m, 'LineWidth', 2); %plot(x, y, '+', 'MarkerSize', 12)
cond(kPost)
[cholpost p] = chol(kPost);

kPost = nearestSPD(kPost);
for i=1:10
    mm = m + (chol(kPost)')*randn(101,1);
    plot(z, mm, 'LineWidth', 1);
end

return

% try to get only the relevant part of the coveriance matrix
% but this gives still non-SPD and functions with small variance
kTT = feval(covfunc{:}, hyp.cov, z(:,2));
kTI = feval(covfunc{:}, hyp.cov, z(:,2),x(:,2));
kII = feval(covfunc{:}, hyp.cov, x);
sigma = exp(2*hyp.lik)*eye(max(size(x))); 
% Posterior covariance matrix
kPost = kTT - kTI/(kII+sigma)*kTI';
kPost = nearestSPD(kPost);
for i=1:10
    mm = m + (chol(kPost)')*randn(101,1);
    %plot(z, mm, 'LineWidth', 1);
end


return 

% set(gca, 'FontSize', 24)
% f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
% disp('fill([z; flipdim(z,1)], f, [7 7 7]/8);')
% fill([z; flipdim(z,1)], f, [7 7 7]/8);

disp('hold on; plot(z, m); plot(x, y, ''+'')')
hold on; plot(z, m, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
for i=1:10
    mm = m + (chol(kPost)')*randn(101,1);
    plot(z, mm, 'LineWidth', 1);
end
axis([-1.9 1.9 -0.9 3.9])
grid on
xlabel('input, x')
ylabel('output, y')
if write_fig, print -depsc f2.eps; end
disp(' '); 


%% sect3
disp(' ')
disp('covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);')
covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
disp('hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y)')
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
disp(' ')

disp('exp(hyp2.lik)')
exp(hyp2.lik)
disp('nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)')
nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)
disp('[m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, z);')
[m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, z);

disp(' ')
figure(3)
set(gca, 'FontSize', 24)
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8)');
fill([z; flipdim(z,1)], f, [7 7 7]/8)
disp('hold on; plot(z, m); plot(x, y, ''+'')');
hold on; plot(z, m, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
grid on
xlabel('input, x')
ylabel('output, y')
axis([-1.9 1.9 -0.9 3.9])
if write_fig, print -depsc f3.eps; end
disp(' '); disp('Hit any key to continue...'); pause

disp(' ')
disp('hyp.cov = [0; 0]; hyp.mean = [0; 0]; hyp.lik = log(0.1);')
hyp.cov = [0; 0]; hyp.mean = [0; 0]; hyp.lik = log(0.1);
disp('hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);')
hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
disp('[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);')
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

figure(4)
set(gca, 'FontSize', 24)
disp(' ')
disp('f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];')
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8)')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
disp('hold on; plot(z, m); plot(x, y, ''+'');')
hold on; plot(z, m, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
grid on
xlabel('input, x')
ylabel('output, y')
axis([-1.9 1.9 -0.9 3.9])
if write_fig, print -depsc f4.eps; end
disp(' '); disp('Hit any key to continue...'); pause

disp('large scale regression using the FITC approximation')
disp('nu = fix(n/2); u = linspace(-1.3,1.3,nu)'';')
nu = fix(n/2); u = linspace(-1.3,1.3,nu)';
disp('covfuncF = {@covFITC, {covfunc}, u};')
covfuncF = {@covFITC, {covfunc}, u};
disp('[mF s2F] = gp(hyp, @infFITC, meanfunc, covfuncF, likfunc, x, y, z);')
[mF s2F] = gp(hyp, @infFITC, meanfunc, covfuncF, likfunc, x, y, z);

figure(5)
set(gca, 'FontSize', 24)
disp(' ')
disp('f = [mF+2*sqrt(s2F); flipdim(mF-2*sqrt(s2F),1)];')
f = [mF+2*sqrt(s2F); flipdim(mF-2*sqrt(s2F),1)];
disp('fill([z; flipdim(z,1)], f, [7 7 7]/8)')
fill([z; flipdim(z,1)], f, [7 7 7]/8)
disp('hold on; plot(z, mF); plot(x, y, ''+'');')
hold on; plot(z, mF, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12)
disp('plot(u,1,''o'')')
plot(u,1,'ko', 'MarkerSize', 12)
grid on
xlabel('input, x')
ylabel('output, y')
axis([-1.9 1.9 -0.9 3.9])
if write_fig, print -depsc f5.eps; end
disp(' ')