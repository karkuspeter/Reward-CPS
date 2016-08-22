function [guess, hyperparams] = PESpredict(Xsamples, Ysamples, guesses, params, hyperparams)
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
    
 	% We update the kernel matrix on the samples
	KernelMatrixInv = {};
	for j = 1 : params.nM
		KernelMatrix = computeKmm(Xsamples, l(j,:)', sigma(j), sigma0(j));
		KernelMatrixInv{ j } = chol2invchol(KernelMatrix);
	end
	f = @(x) posteriorMean(x, Xsamples, Ysamples, KernelMatrixInv, l, sigma);
	gf = @(x) gradientPosteriorMean(x, Xsamples, Ysamples, KernelMatrixInv, l, sigma);

	% We optimize the posterior mean of the GP
	guess = globalOptimization(f, gf, params.xmin, params.xmax, guesses);
    
    hyperparams.l = l;
    hyperparams.sigma = sigma;
    hyperparams.sigma0 = sigma0;
end
