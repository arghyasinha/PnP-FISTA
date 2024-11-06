function [E] = compression_basis(X,basis_type,p)
%COMPRESSION_BASIS Compute PCA/VCA basis for pixels from input image

X = im2mat(X);

switch  basis_type 
    case 'VCA'
%     Find endmembers with VCA (pick the one with smallest volume from 20 
%     runs of the algorithm)
    max_vol = 0;
    vol = zeros(1, 20);
    for idx_VCA = 1:20
        E_aux = VCA(X,'Endmembers',p,'SNR',0,'verbose','off');
        vol(idx_VCA) = abs(det(E_aux'*E_aux));
        if vol(idx_VCA) > max_vol
            E = E_aux;
            max_vol = vol(idx_VCA);
        end
    end
    case 'SVD'
%     Learn the subspace with SVD
    [E, ~] = svds(X,p);

end

end

