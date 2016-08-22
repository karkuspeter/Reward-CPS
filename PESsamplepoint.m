function [Xsample, hyperparams] = PESsamplepoint(Xsamples, Ysamples, guesses, params, hyperparams)
    %only provide hyperparams if it is already optimized for Xsamples Ysamples

    if nargin < 5
        % We sample from the posterior distribution of the hyper-parameters
        [ l, sigma, sigma0 ] = sampleHypers(Xsamples, Ysamples, params.nM);
        hyperparams = struct();
    else
        l = hyperparams.l;
        sigma = hyperparams.sigma;
        sigma0 = hyperparams.sigma0;
    end
        
    % We sample from the global minimum
	[ m hessians ] = sampleMinimum(params.nM, Xsamples, Ysamples, sigma0, sigma, l, params.xmin, params.xmax, params.nFeatures);

	% We call the ep method
	ret = initializeEPapproximation(Xsamples, Ysamples, m, l, sigma, sigma0, hessians);

	% We define the cost function to be optimized
	cost = @(x) evaluateEPobjective(ret, x);

	% We optimize globally the cost function
	Xsample = globalOptimizationOneArgument(cost, params.xmin, params.xmax, guesses);	

    hyperparams.l = l;
    hyperparams.sigma = sigma;
    hyperparams.sigma0 = sigma0;
end