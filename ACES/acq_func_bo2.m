function val = acq_func_bo2(GP_cell, Xval, kappa)
  val = 0;
  
  for i=1:length(GP_cell)
      GPrel = GP_cell{i};
      [y, std2] = gp(GPrel.hyp, [], [], GPrel.covfunc, GPrel.likfunc, GPrel.x, GPrel.y, Xval);

      val = val + (y - kappa * sqrt(std2));
  end
end
