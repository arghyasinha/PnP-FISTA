function [probs] = GMM_posteriors(gm,X)
%GMM_POSTERIORS Calculate posterior label probabilities for a GMM
%distribution at specific points

d = size(X,1);          % Feature dimension
N = size(X,2);          % No. of input vectors
K = gm.NumComponents;   % No. of GMM components

if(length(size(gm.Sigma))==3 && size(gm.Sigma,1)==d)
    cov_type = 'full';
elseif(length(size(gm.Sigma))==3 && size(gm.Sigma,1)==1)
    cov_type = 'diag';
elseif(length(size(gm.Sigma))==2 && size(gm.Sigma,1)==d)
    cov_type = 'full_common';
elseif(length(size(gm.Sigma))==2 && size(gm.Sigma,1)==1)
    cov_type = 'diag_common';
end

probs = zeros(K,N);
for k = 1:K
    wk = gm.ComponentProportion(k); % Weight of kth component
    if(strcmp(cov_type,'full'))
        C_k = gm.Sigma(:,:,k);
    elseif(strcmp(cov_type,'diag'))
        C_k = diag(gm.Sigma(:,:,k));
    elseif(strcmp(cov_type,'full_common'))
        C_k = gm.Sigma;
    elseif(strcmp(cov_type,'diag_common'))
        C_k = diag(gm.Sigma);
    end
    fk = gauss(X,gm.mu(k,:)',C_k);    % Density value of kth component at X
    probs(k,:) = wk*fk;
end
denom = sum(probs,1);
probs = probs./denom;   % Normalize

end

function f = gauss(X,m,C)

d = size(X,1);

Y = C\(X-m);
arg = sum((X-m).*Y,1);
num = exp(-arg/2);
denom = sqrt(det(C)*((2*pi)^d));
f = num./denom;

end
