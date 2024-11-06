
clearvars; close all; clc;
addpath('./utils');

load('./HySure-master/data/original_rosis.mat','X');
load('./HySure-master/data/ikonos_spec_resp.mat','ikonos_sp');
load('./GMM_rosis.mat','GMmodel');

patch_len = 8;      % Patch length
stride = 1;         % Patch stride
sigma_n = 15/255;   % Noise level (for testing denoising ability)

[~,I_ms,I_pan] = forward_model(X,4,30,40,50,ikonos_sp);

I_ground = I_ms(:,:,1);
patches_clean = im2patches(I_ground,[patch_len,patch_len],[stride,stride]);

% I_pan_noisy = I_pan + sigma_n*randn(size(I_pan));
I_noisy = I_ground + sigma_n*randn(size(I_ground));
[patches_noisy,rowinds,colinds] = im2patches(I_noisy,[patch_len,patch_len],[stride,stride]);

label_probs = GMM_posteriors(GMmodel,patches_clean);
denoised_patches = GMM_MMSE(GMmodel,patches_noisy,sigma_n,label_probs);
J = patches2im(denoised_patches,rowinds,colinds,[patch_len,patch_len],size(I_ground));

% addpath('./BM3D/');
% [~,Jbm3d] = BM3D(I_ground,I_noisy,sigma_n*255,'np',0);

fprintf('PSNR (GMM) = %f\n',psnr(J,I_ground,1));
% fprintf('PSNR (BM3D) = %f\n',psnr(Jbm3d,I_ground,1));

figure;
subplot(1,3,1); imshow(I_noisy); title('Noisy');
subplot(1,3,2); imshow(J); title('GMM denoised');
% subplot(1,3,3); imshow(Jbm3d); title('BM3D denoised');
