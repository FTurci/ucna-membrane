# implementing PHYSICAL REVIEW E 96, 032601 (2017)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps
from scipy.interpolate import UnivariateSpline


def potential(x,d=1,epsilon=10, reassign=True):
  return np.piecewise(x, [np.absolute(x)<d, np.absolute(x)>d],[lambda s:epsilon*(np.cos(np.pi*s/d)+1),lambda s:0*s] )
#   if reassign:
#     d = 1/epsilon
    
#   return epsilon*(np.tanh((x+1)/d)-np.tanh((x-1)/d))

  
class UCNAprofile:
  def __init__(self,xmin, xmax,zeta=1, npoints=1000, d=1,s=0.001,epsilon=10.0):
    self.x = np.linspace(xmin,xmax, npoints)
    self.xmin = xmin
    self.xmax = xmax
    self.npoints = npoints
    self.zeta = zeta
    self.epsilon = epsilon
    pot = potential(self.x,d,epsilon)
    
    self.U = UnivariateSpline(self.x,pot, s=s, k=3)
    self.dUx = self.U.derivative()
    self.dUxx = self.dUx.derivative()
    
    self.U_ = self.U(self.x)
    self.dUx_ = self.dUx(self.x)
    self.dUxx_ = self.dUxx(self.x)
    
    self.g = 1+1/zeta**2*self.dUxx_
    

  def plot_potential(self,ax=None,derivatives=True, legend=True):
    if ax==None:
      a = plt.gca()
    else:
      a = ax
    a.plot(self.x, self.U(self.x), label = 'U')
    
    if derivatives:
      a.plot(self.x, self.dUx(self.x), label = 'U\'')
      a.plot(self.x, self.dUxx(self.x), label = 'U\'\'')
    if legend:
      a.legend()
  
  
  def effective_potential_at_y(self,y):
    valid =self.x<=y
    
    a = 1+self.dUxx_[valid]/self.zeta
    b = self.dUx_[valid]
    return simps(a*b,self.x[valid])
  
  def effective_potential(self,y):
    return np.vectorize(self.effective_potential_at_y)(y)