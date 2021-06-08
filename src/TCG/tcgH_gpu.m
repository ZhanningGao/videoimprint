function [pi,pl,Lq,loglikelihood_samples,loglikelihood,H] = tcgH_gpu( counts, E, W, S, options )
% [pi,pl,Lq,loglikelihood_samples] =  cg( counts, E, W, S, options )
%
% 2-dimensional Counting Grids - UAI 2011
%
% ---- INPUT ----
% counts:            data, features x samples
% E:                 counting grid size  (e.g., [30,30])
% W:                 Ws's window size (e.g., [4/S 4/S])
% S:                 tessellated sectors
% options (Struct) - All fields are optional.
%   min_change:      convergence criterion. Min relative change in
%                    loglikelihood (defalut 1e-4)
%   max_iter:        max number of EM iteration
%   learn_pi:        learn counting grid \pi (default 1=true)
%   learn_pl:        learn prior on locations P( \bf{k} ) (default 0=false)
%   plot_figure:     plot the loglikelihood after every iteration
%   normalize_data:  normalize the data (default 1=true, SET TO ZERO FOR
%                    TEXT!)
%   pi:              Learned Counting Grid (for inference, or to pass an
%                    initialization)
%   pl:              Learned Prior (for inference, or to pass an
%                    initialization)
%
% ---- OUTPUT ----
% pi:                counting grid pi(\bf{k},z) = \pi_{\bf{k},z}
% pl:                prior on locations P(i,j)
% Lq:                location posterior for each sample log p(\bf{k}| c^t )
% loglikelihood_samples: loglikelihood of each sample
%
%
% Written by Alessandro Perina, alessandro.perina@gmail.com /
% alperina@microsoft.com
% Modified by Zhanning Gao 2015.11.30 for Tessellated CG

dbstop if error

if ~exist( 'options', 'var'); options = []; end
if ~isfield( options,'min_change'); options.min_change = 1e-4; end
if ~isfield( options,'max_iter'); options.max_iter = 100; end
if ~isfield( options,'learn_pi'); options.learn_pi = 1; end
if ~isfield( options,'learn_pl'); options.learn_pl = 0; end
if ~isfield( options,'plot_figure'); options.plot_figure = 0; end
if ~isfield( options,'normalize_data'); options.normalize_data = 1; end

[Zall,T]=size(counts);
%countsUint = uint32(counts);
Z = Zall/S/S;
if options.normalize_data
      for s = 1:S*S
          tmp_counts = counts((s-1)*Z+1:s*Z,:);
          tmp_counts(tmp_counts<0) = 0;
          %counts((s-1)*Z+1:s*Z,:) = 100*prod(W*S*S)*bsxfun( @rdivide, tmp_counts, sum(tmp_counts,1) );
          %countsUint((s-1)*Z+1:s*Z,:) = uint32(100*prod(W*S*S)*bsxfun( @rdivide, tmp_counts, sum(tmp_counts,1) ));
          counts((s-1)*Z+1:s*Z,:) = 100*prod(W*S*S)*(yael_vecs_normalize(tmp_counts).^2);
      end
  %  counts=100*prod(W*S)*bsxfun( @rdivide, counts, sum(counts,1) );
    %counts=100*prod(W*S)*(yael_vecs_normalize(counts).^2);

end

L = prod(E);
total = sum(counts(:));

%counts = single(countsUint);
if isfield( options,'pi')
    pi = options.pi;
else
    pi = gpuArray(1+1*rand([E,Z]));
    pi = bsxfun( @rdivide, pi, sum(pi,3));
end

if isfield( options,'pl');
    pl = options.pl;
else
    pl = gpuArray(ones(E)/L);
end

PI = padarray( ...
    permute(cumsum( permute( cumsum( ...
    padarray(pi,W-1,'circular','post' )),[2 1 3]) ),[2 1 3] ), ...
    [1 1],0,'pre');
tmp = compute_h_noLoopFull( PI, W(2)+1, W(1)+1);
h{1} = bsxfun( @rdivide, tmp(1:end,1:end,:), sum( tmp(1:end,1:end,:),3 ));

for s = 2:S*S
    sr = ceil(s/S);
    sc = s - (sr-1)*S;
    h{s} = circshift(h{1},[-(sr-1)*W(1), -(sc-1)*W(2)]);
end


iter = 1;
alpha = 1e-10;
start_iterating_m = 1; % Start M-step iterations from
m_step_iter = 1; % M-step iterations: fasten convergence


if isfield( options,'pi')&&isfield(options, 'alphaPi')
    pseudocounts = options.alphaPi*pi;
else
    pseudocounts =  mean( sum(counts) / prod(E) )  / 2.5;
end

converged = iter > options.max_iter;
loglikelihood = gpuArray(zeros(1,options.max_iter));
loglikelihood_samples = gpuArray(zeros(1,T));
minp = 1/(10*L);
Lq = gpuArray(zeros([E,T]));
while ~converged
    lql = gpuArray(zeros([L,T]));
    for s = 1:S*S
        
        if options.learn_pl
            lql = lql + bsxfun(@plus, reshape( log( pl ),[L,1]), reshape( log( h{s}), [L,Z])*counts((s-1)*Z+1:s*Z,:) );
        else
            lql = lql + reshape( log( h{s}), [L,Z])*counts((s-1)*Z+1:s*Z,:);
        end
    end
    Lq = reshape( bsxfun( @minus, bsxfun(@minus, lql, max(lql) ),  log( sum( exp( bsxfun(@minus, lql, max(lql) ) ) ))), [E,T]);
    %    Lq = reshape( bsxfun(@minus, lql, max(lql) ), [E,T]);
    tmp = exp( Lq ) ;  tmp( tmp< minp ) = minp;   Lq = log( bsxfun(@rdivide, tmp, sum( sum(tmp)) ));
    
    
    miter = 1;
    if iter > start_iterating_m
        miter = m_step_iter;
    end
    
    for int_it = 1:miter
        
        if options.learn_pi
            QH_sum = gpuArray(zeros([E,Z]));
            for s = 1:S*S
                sr = ceil(s/S);
                sc = s - (sr-1)*S;
                nrm = reshape(  reshape( padarray( exp( Lq), W*S-1, 'circular','pre'), [prod(E+W*S-1),T])*counts((s-1)*Z+1:s*Z,:)', [ E+W*S-1,Z ]);
                
                QHtmp = padarray(permute( cumsum( permute( cumsum(...
                    bsxfun( @rdivide, nrm,   padarray(h{s}+prod(W)*alpha ,[W*S-1,0],'circular','pre')) ), ...
                    [2 1 3]) ),[2 1 3]),...
                    [1 1],0,'pre');
                tmp = compute_h_noLoopFull(QHtmp,W(2)+1,W(1)+1);
                QH = tmp(1:end,1:end,:);
                QH(QH<0) = 0;
                QH_sum = QH_sum + QH(1+(S-sr)*W(2):E(1)+(S-sr)*W(2),1+(S-sc)*W(1):E(2)+(S-sc)*W(1),:);
                %     QH_sum = QH_sum + QH(1+(sr-1)*W(2):E(1)+(sr-1)*W(2),1+(sc-1)*W(1):E(2)+(sc-1)*W(1),:);
            end
            
            
            un_pi = pseudocounts  + QH_sum.*(pi+alpha);
            
            
            mask = sum(un_pi,3) ~= 0;
            
            pi = bsxfun(@times, bsxfun(@rdivide, un_pi, sum(un_pi,3)), double( mask ) ) ...
                + bsxfun(@times, 1/Z*ones([E,Z]), double( ~mask ) );
            
            PI = padarray( ...
                permute(cumsum( permute( cumsum( ...
                padarray(pi,W-1,'circular','post' )),[2 1 3]) ),[2 1 3] ), ...
                [1 1],0,'pre');
            tmp = compute_h_noLoopFull( PI, W(2)+1, W(1)+1);
            h{1} = bsxfun( @rdivide, tmp(1:end,1:end,:), sum( tmp(1:end,1:end,:),3 ));
            for s = 2:S*S
                sr = ceil(s/S);
                sc = s - (sr-1)*S;
                h{s} = circshift(h{1},[-(sr-1)*W(1), -(sc-1)*W(2)]);
            end
        end
        
        if options.learn_pl
            msk = padarray( ones(W),E-W,0,'post');
            pl = zeros(E);
            for t=1:T
                tmp = real( ifft2( fft2( msk ).*fft2( exp(Lq(:,:,t)))) );
                tmp( tmp > 1 ) = 1; tmp( tmp<0) = 0;
                pl = pl + tmp;
            end
            pl = pl ./ sum(pl(:));
        end
        
    end
    loglikelihood_samples = gpuArray(zeros([1,T]));
    for s=1:S*S
        loglikelihood_samples = loglikelihood_samples + sum( reshape( exp(Lq),[L,T]).*( reshape( log( h{s}), [L,Z])*counts((s-1)*Z+1:s*Z,:) ));
    end
    loglikelihood_samples = loglikelihood_samples - squeeze( sum(sum( exp(Lq).*Lq )))';
    loglikelihood(iter) = sum( loglikelihood_samples );
    
    
    if options.plot_figure == 1;
        figure(1), plot(1:iter, loglikelihood(1:iter),'.-r'); grid on; drawnow;
        title('Data loglikelihood');
        %         for r = 1:E(1)
        %             for c = 1:E(2)
        %                 re_img(r,c,:) = sum(repmat(reshape(pi(r,c,:),[size(options.codebooks,1),1]),[1,3]).*options.codebooks);
        %             end
        %         end
        %figure(1);imshow(imresize(padarray(re_img,[33,40,0],'circular','post'),[330,400],'nearest'));title(num2str(iter));drawnow;
        figure(2);imagesc(sum(gather(Lq),3));drawnow;
        %pause(0.2);
    end
    
    converged = iter >= options.max_iter;
    if iter > 50
        F1 = loglikelihood(iter)/total;
        F2 = loglikelihood(iter-1)/total;
        rel_ch = (F1-F2) / abs(mean([F1,F2]));
        if rel_ch < options.min_change
            converged = 1;
        end
    end
    iter = iter+1;
    
end

H = h{1};

end

function h = compute_h_noLoopFull( H, xW, yW )
h = H(yW:end,xW:end,:,:,:) - H(1:end-yW+1,xW:end,:,:,:) ...
    - H(yW:end,1:end-xW+1,:,:,:) + H(1:end-yW+1,1:end-xW+1,:,:,:);
end