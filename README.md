# FFT-based Homogenization

![githubFFT](https://user-images.githubusercontent.com/52761240/61006553-e7a93e00-a36a-11e9-8db7-56725e06e296.png)

This code is an implementation of the standard FFT homogenization method developed by Moulinec and Suquet (1998) for microstructures with linear elastic isotropic phases.
It computes the effective (/homogenized) stiffness of a microstructure with two phases (in the code the inclusion has the shape of a cylinder).

By now, many different FFT homogenization methods have been developded, see for example

[1] J. Zeman, T.W.J. de Geus, J. Vondřejc, R.H.J. Peerlings, M.G.D. Geers. A finite element perspective on non-linear FFT-based micromechanical simulations. 
International Journal for Numerical Methods in Engineering, Accepted, 2016. [arXiv: 1601.05970](https://arxiv.org/abs/1601.05970).
[2] T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings, M.G.D. Geers. Finite strain FFT-based non-linear solvers made simple. Submitted, 2016. arXiv: [1603.08893](https://arxiv.org/abs/1603.08893).

See also: https://github.com/tdegeus/GooseFFT where the basis of the codes in this repositoy can be found.
