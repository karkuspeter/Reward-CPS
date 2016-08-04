function val = acq_func_bo(gprMdl, Xval, kappa)
  [y, std] = gprMdl.predict(Xval);
  val = -(y + kappa * std);
end
