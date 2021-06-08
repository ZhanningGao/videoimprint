%get QL with reference pi and its active map Amap

function [PI, QL, Amap, H] = getQLviaPI_gpu(feature, pi, E, W, S, ifLearnPi, alphaPi, amap_t)

%options.pi = pi;
options.learn_pi=ifLearnPi;
options.alphaPi=alphaPi;
options.plot_figure = 0;

[PI,~,Lq,~,~,H] = tcgH_gpu( gpuArray(feature), [E,E], [W/S,W/S], S, options );

Amap = zeros(E,E);

%QL = exp(Lq);
QL = single(exp(gather(Lq)));
H = single(gather(H));
PI = single(gather(PI));
%Amap(sum(QL,3)>amap_t)=1;