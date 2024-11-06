function [X_hat] = GMM_MMSE(gm,Y,sigma_n,posteriors)
%GMM_MMSE Calculate the MMSE estimates of a given set of noisy data samples
%assuming a GMM prior and Gaussian noise
% gm = Fitted gmdistribution model containing GMM info
% Y = d-by-N set of noisy samples, where d = feature dimension and N = no.
% of samples (each column represents a samples)
% sigma_n = Noise standard deviation
% posteriors = K-by-N matrix of posterior probabilities of the samples,
% where K = no. of GMM components. If empty, they will be calculated from
% the noisy samples Y. The main utility of this parameter is to supply a
% set of posterior densities calculated from the corresponding clean
% samples, which is possible in some applications such as hyperspectral
% pan sharpening.

if(~exist('posteriors','var') || isempty(posteriors))
    posteriors = GMM_posteriors(gm,Y);
end

K = gm.NumComponents;
d = size(Y,1);
N = size(Y,2);

if(length(size(gm.Sigma))==3 && size(gm.Sigma,1)==d)
    cov_type = 'full';
elseif(length(size(gm.Sigma))==3 && size(gm.Sigma,1)==1)
    cov_type = 'diag';
elseif(length(size(gm.Sigma))==2 && size(gm.Sigma,1)==d)
    cov_type = 'full_common';
elseif(length(size(gm.Sigma))==2 && size(gm.Sigma,1)==1)
    cov_type = 'diag_common';
end

sigma2I = sigma_n*sigma_n*eye(d);
X_hat = zeros(d,N);
for k = 1:K
    if(strcmp(cov_type,'full'))
        C_k = gm.Sigma(:,:,k);
    elseif(strcmp(cov_type,'diag'))
        C_k = diag(gm.Sigma(:,:,k));
    elseif(strcmp(cov_type,'full_common'))
        C_k = gm.Sigma;
    elseif(strcmp(cov_type,'diag_common'))
        C_k = diag(gm.Sigma);
    end
    V_k = (C_k + sigma2I)\Y;
    V_k = C_k * V_k;
    X_hat = X_hat + repmat(posteriors(k,:),d,1) .* V_k;
end

end

