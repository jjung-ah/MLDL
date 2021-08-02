def batchnorm_backward(dout, cache):

	  #unfold the variables stored in cache
	  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
	
	  #get the dimensions of the input/output
	  N,D = dout.shape
	
	  #step9
	  dbeta = np.sum(dout, axis=0)
	  dgammax = dout #not necessary, but more understandable
	
	  #step8
	  dgamma = np.sum(dgammax*xhat, axis=0)
	  dxhat = dgammax * gamma
	
	  #step7
	  divar = np.sum(dxhat*xmu, axis=0)
	  dxmu1 = dxhat * ivar
	
	  #step6
	  dsqrtvar = -1. /(sqrtvar**2) * divar
	
	  #step5
	  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
	
	  #step4
	  dsq = 1. /N * np.ones((N,D)) * dvar
	
	  #step3
	  dxmu2 = 2 * xmu * dsq
	
	  #step2
	  dx1 = (dxmu1 + dxmu2)
	  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
	
	  #step1
	  dx2 = 1. /N * np.ones((N,D)) * dmu
	
	  #step0
	  dx = dx1 + dx2
	
	  return dx, dgamma, dbeta