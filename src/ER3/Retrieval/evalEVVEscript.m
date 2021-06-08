% test Retrival on EVVE

addpath ..\..\utils
load('..\..\..\data\GndEVVE.mat');
load('..\..\..\data\PCAw_CGres5.mat');
load('..\..\..\data\video_imprint\cnnCG.mat');

[averagemAP, ~] = evalEVVE(cnnGCG, queryEVVE, eventGND, meanRes5, 'CG', 'COS', PCAw_res5);