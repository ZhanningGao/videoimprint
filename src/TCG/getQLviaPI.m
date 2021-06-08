%get QL with reference pi and its active map Amap

function [PI, QL, Amap, H] = getQLviaPI(feature, pi, E, W, S, ifLearnPi, alphaPi, amap_t)

%options.pi = pi;
options.learn_pi=ifLearnPi;
options.alphaPi=alphaPi;
options.plot_figure = 1;

[PI,~,Lq,~,~,H] = tcgH( feature, [E,E], [W,W], S, options );

Amap = zeros(E,E);

QL = exp(Lq);
Amap(sum(QL,3)>amap_t)=1;