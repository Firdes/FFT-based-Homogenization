import numpy as np
import scipy.sparse.linalg as sp
import itertools
from mayavi import mlab	#scientific visualization library

################# STANDARD FFT-HOMOGENIZATION by Moulinec & Suquet (1994) ###############

# ----------------------------------- GRID ------------------------------------

ndim   = 3            # number of dimensions
N      = 31           # number of voxels (assumed equal for all directions, needs to be an odd number)
ndof   = ndim**2*N**3 # number of degrees-of-freedom

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
trans2 = lambda A2 : np.einsum('ijxyz          ->jixyz  '     ,A2)
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([N,N,N]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([N,N,N]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([N,N,N]))
I4s    = (I4+I4rt)/2. # symm. 4th order tensor
II     = dyad22(I,I)  # dyadic product of 2nd order unit tensors

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction 
# here: inclusion has cylinder form
phase  = np.zeros([N,N,N])
r = np.sqrt((0.2*N**2)/np.pi) #radius of cylinder (20% volume fraction)
for i in range(N):
	for j in range(N):
		for k in range(N):
			if ((i-int(N/2))**2 + (k-int(N/2))**2) <= r:
				phase[i,j,k]=1.

## Visualization with Mayavi
# X, Y, Z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
# mlab.points3d(X, Y, Z, phase, color=(0.86, 0.24, 0.22), scale_factor=0.1, mode='cube')
# mlab.outline(color=(0.24, 0.56, 0.71), line_width=2.7)
# mlab.show()

# material parameters + function to convert to grid of scalars
param   = lambda M1,M2: M1*np.ones([N,N,N])*(1.-phase)+M2*np.ones([N,N,N])*phase
lambda1 = 10.0
lambda2 = 100.0
lambdas = param(lambda1, lambda2)  # Lamé constants (material1, material2)  [grid of scalars]
mu1     = 0.25
mu2 	= 2.5
mu      = param(mu1, mu2)  	       # shear modulus [grid of scalars]
## stiffness tensor                [grid of scalars]  
C4      = lambdas*II+2.*mu*I4s 

# ------------------------------------------------------------------------------

## projection operator                            [grid of tensors]
delta  	= lambda i,j: np.float(i==j)              # Dirac delta function
freq   	= np.arange(-(N-1)/2.,+(N+1)/2.)          # coordinate axis -> freq. axis
lambda0 = (lambda1 + lambda2)/2 				  # Lamé constant for isotropic reference material
mu0     = (mu1 + mu2)/2           				  # shear modulus for isotropic reference material
const  	= (lambda0 + mu0)/(mu0*(lambda0 + 2*mu0))
Greens  = np.zeros([ndim,ndim,ndim,ndim,N,N,N])   # Green's function in Fourier space

for k,h,i,j in itertools.product(range(ndim), repeat=4):
    for x,y,z in itertools.product(range(N), repeat=3):
        q = np.array([freq[x], freq[y], freq[z]]) # frequency vector
        if not q.dot(q) == 0: # zero freq. -> mean
            Greens[k,h,i,j,x,y,z] = (1/(4*mu0*q.dot(q))*\
             (delta(k,i)*q[h]*q[j]+delta(h,i)*q[k]*q[j]+\
              delta(k,j)*q[h]*q[i]+delta(h,j)*q[k]*q[i]))-\
              (const*((q[i]*q[j]*q[k]*q[h])/(q.dot(q))**2))

# (inverse) Fourier transform (for each tensor component in each direction)
fft  = lambda x: np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))

# inner functions to produce IB matrix, IB = I - F^{-1} *Gamma*F*C --> eps_i+1 = IB*eps_i 
G           = lambda x: np.real(ifft(ddot42(Greens,fft(x)))).reshape(-1)
Stiff_Mat   = lambda x: ddot42(C4,x.reshape(ndim,ndim,N,N,N))
G_Stiff_Mat = lambda x: G(Stiff_Mat(x))
Id 		    = lambda x: ddot42(I4,x.reshape(ndim,ndim,N,N,N)).reshape(-1)
IB 		    = lambda x: np.add(Id(x),-1.*G_Stiff_Mat(x))

# # ----------------------------- NEWTON ITERATIONS -----------------------------

# initialize stress and strain for each 6 macroscopic strain E  
sig = [np.zeros([ndim,ndim,N,N,N]) for _ in range(6)]
eps = [np.zeros([ndim,ndim,N,N,N]) for _ in range(6)]

# set macroscopic strains, total:6 (for each direction)
E   = [np.zeros([ndim,ndim,N,N,N]) for _ in range(6)]

E[0][0][0] = 1.0 # loading in 1,1 direction 
E[1][1][1] = 1.0 # loading in 2,2 direction
E[2][2][2] = 1.0 # loading in 3,3 direction
E[3][0][1] = 1.0 # loading in 1,2 direction
E[3][1][0] = 1.0 # loading in 2,1 direction (due to symmetry)
E[4][1][2] = 1.0 # loading in 2,3 direction
E[4][2][1] = 1.0 # loading in 3,2 direction (due to symmetry)
E[5][0][2] = 1.0 # loading in 1,3 direction
E[5][2][0] = 1.0 # loading in 3,1 direction (due to symmetry)

iiter = [0 for _ in range(6)]

# --------------- for convergence criteria ----------------

freqMat = np.zeros([ndim, 1, N, N, N])		#[grid of scalars]
for j in range(ndim):
	for x in range(N):
		for y in range(N):
			for z in range(N):
				if j==0:
					freqMat[j,0,x,y,z] = freq[x]
				elif j==1:
					freqMat[j,0,x,y,z] = freq[y]
				elif j==2:
					freqMat[j,0,x,y,z] = freq[z] 

freqMat_T = trans2(freqMat)
c 		  = int((N-1)/2) 					# center of frequency grid
# ---------------------------------------------------------

for i in range(6):
	sigma = np.zeros([ndim,ndim,N,N,N])
	eps[i] += E[i]
	while True:
		eps[i] 	 = IB(eps[i])
		sigma    = Stiff_Mat(eps[i])

		# ---------------- (equilibrium-based) convergence criteria -------------------------
		fou_sig  = fft(sigma).reshape(ndim, ndim, N,N,N)
		nom 	 = np.sqrt(np.mean(np.power(dot22(freqMat_T, fou_sig),2))) #nominator
		denom 	 = np.sqrt(np.real(fou_sig[0,0,c,c,c]**2 + fou_sig[1,1,c,c,c]**2 +\
					fou_sig[2,2,c,c,c]**2 + fou_sig[0,1,c,c,c]**2 +\
					fou_sig[1,2,c,c,c]**2 + fou_sig[0,2,c,c,c]**2)) # denominator
		# ---------------------------------------------------------------
		
		if nom/denom <1.e-8 and iiter[i]>0: break
		iiter[i] += 1

# homogenized stiffness
homStiffness = np.zeros([6, 6])


X, Y, Z = np.mgrid[0:1:N*1j, 0:1:N*1j, 0:1:N*1j]
mlab.points3d(X, Y, Z, eps[0][0][0], color=(0.86, 0.24, 0.22), scale_factor=0.1, mode='cube')
mlab.scalarbar(title="strain", orientation='vertical')
mlab.outline(color=(0.24, 0.56, 0.71), line_width=2.7)
mlab.show()


# homogenization operation <f> = 1/N Σ f_i
for i in range(6):
	sig[i] = Stiff_Mat(eps[i])
	homStiffness[0][i] = round((1.0/(N**3))*np.sum(sig[i][0][0]),4)
	homStiffness[1][i] = round((1.0/(N**3))*np.sum(sig[i][1][1]),4)
	homStiffness[2][i] = round((1.0/(N**3))*np.sum(sig[i][2][2]),4)
	homStiffness[3][i] = round((1.0/(N**3))*np.sum(sig[i][0][1]),4)
	homStiffness[4][i] = round((1.0/(N**3))*np.sum(sig[i][1][2]),4)
	homStiffness[5][i] = round((1.0/(N**3))*np.sum(sig[i][2][0]),4)

print("Homogenized Stiffness: \n", homStiffness)
