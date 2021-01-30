# ML_asset_management
Implementing code from Machine Learning for Asset Managers by Lopez de Prado

This repo reproduces the code found in Machine Learning for Asset Managers by Lopez de Prado, with added comments and code for my personal use and understanding.

marcenko_pastur.py
------------------
This piece of code implements the Marcenko-Pastur theorem for calculating the
pdf of a matrix's eigenvalues.
We check that the theorem works by using simulated random data and fitting a kernel density estimator. We then plot the two approaches and see that they coincide.
* mpPDF: takes in var, q, pts, returns the Marcenko-Pastur pdf for eigenvalues
var is the variance of the data generating process
q = T/N where we are looking at the eigenvalues of matrix C = T^-1*X'X where X
is TxN
pts is how granular the linspace over which we define the pdf is, concretely the number of eigenvalues.
* getPCA: takes in a matrix, and returns the eigenvalues and eigenvectors in a 
usable format.
* fitKDE: takes in observations and KDE arguments, returns the pdf from a KDE fit
over these observations
Fit the KDE with the observations, then keep unique observations in variable x 
to draw the pdf

The remainder of the code uses the MP theorem and the KDE approach to compute
a test pdf over synthetic data, and then plots the results.

random_mat_with_signal.py
------------------