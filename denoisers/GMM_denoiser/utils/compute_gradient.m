function grad = compute_gradient(X,SB,SBadj,E,R,Yh,Ym,lambda)
%COMPUTE_GRADIENT Gradient of data fidelity term for HS-MS image fusion
% X = Input image at which gradient is to be computed
% SB = Handle to function that performs spatial degradation (blurring and
% downsampling)
% SBadj = Handle to function that performs the adjoint operation
% (upsampling and blurring)
% E = Matrix of PCA/VCA basis vectors
% R = Spectral degradation matrix
% Yh = Observed hyperspectral image
% Ym = Observed multispectral image
% lambda = Regularization parameter which decides relative weightage of
% spectral degradation objective

term1 = SB(X);
term1 = SBadj(term1);
term1 = E'*E*term1;

term2 = lambda * (R*E)' * (R*E) * X;

term3 = SBadj(Yh);
term3 = (E')*term3;

term4 = lambda * (R*E)' * Ym;

grad = term1 + term2 - (term3 + term4);

end

