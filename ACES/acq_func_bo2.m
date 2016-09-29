function val = acq_func_bo2(GPrel, Xval, kappa)
      [y, std2] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y, Xval);
      val = (y - kappa * sqrt(std2));
end
