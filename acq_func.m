function val = acq_func(gprMdl, Xval, kappa)
  [y, std] = gprMdl.predict(Xval);
  val = -(y + kappa * std);
end
