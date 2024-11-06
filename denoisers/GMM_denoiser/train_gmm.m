
load('./HySure-master/data/original_rosis.mat','X');
load('./HySure-master/data/ikonos_spec_resp.mat','ikonos_sp');
addpath('./utils/');

p_len = 8;      % Patch length
K = 20;         % No. of GM components
savefile = './GMM_rosis.mat';   % File to save the trained GMM

[~,~,I_train] = forward_model(X,4,30,40,50,ikonos_sp);

patches = im2col(I_train,[p_len,p_len]);
patches = patches';

GMmodel = fitgmdist(patches,K);
save(savefile,'GMmodel');
