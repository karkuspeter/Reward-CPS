
addpath ../../PES-NIPS2014/GPstuff-4.4/diag
addpath ../../PES-NIPS2014/GPstuff-4.4/dist
addpath ../../PES-NIPS2014/GPstuff-4.4/gp
addpath ../../PES-NIPS2014/GPstuff-4.4/mc
addpath ../../PES-NIPS2014/GPstuff-4.4/misc
addpath ../../PES-NIPS2014/GPstuff-4.4/optim
%addpath ../../PES-NIPS2014/GPstuff-4.4/xunit
addpath ../../PES-NIPS2014/sourceFiles


% the objective function to be minmized
objective = @(x) target(x);

% The number of samples from the global optimum to be drawn, the boundaries and the number of random features
PESparams = struct(...
    'nM', 200, ...
    'xmin', [ 0 ; 0 ], ...
    'xmax', [ 1 ; 1 ], ...
    'nFeatures', 1000 ...
    );

% We initialize the random seed
%s = RandStream('mcg16807','Seed', simulation * 10000);
%RandStream.setGlobalStream(s);

% We obtain three random samples

nInitialSamples = 3;
Xsamples = lhsu(xmin, xmax, nInitialSamples);
guesses = Xsamples;
Ysamples = zeros(nInitialSamples, 1);
r_guesses = zeros(nInitialSamples, 1);
for i = 1 : nInitialSamples
	Ysamples(i) = objective(Xsamples(i,:));
end
start = nInitialSamples + 1;

% We sample from the posterior distribution of the hyper-parameters
%[ l, sigma, sigma0 ] = sampleHypers(Xsamples, Ysamples, nM);



for i = start : 5
	fprintf(1, '%d\n', i);
    
    [Xsamplepoint h] = PESsamplepoint(Xsamples, Ysamples, guesses, PESparams);
    
  	% We collect the new measurement
	Xsamples = [ Xsamples ; Xsamplepoint ];
	Ysamples = [ Ysamples ; objective(Xsamplepoint) ];

    [guess h] = PESpredict(Xsamples, Ysamples, guesses, PESparams);
    
    guesses = [ guesses ; guess ];
    r_guesses = [r_guesses; objective(guess)];
end

showtarget