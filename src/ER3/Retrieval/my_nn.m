% Return the k nearest neighbors of a set of query vectors
%
% Usage: [ids,dis] = nn(v, q, k, distype)
%   v                the dataset to be searched (one vector per column)
%   q                the set of queries (one query per column)
%   k  (default:1)   the number of nearest neigbors we want
%   distype          distance type: 1=L1,
%                                   2=L2         -> Warning: return the square L2 distance
%                                   3=chi-square -> Warning: return the square Chi-square
%                                   4=signed chis-square
%                                   16=cosine    -> Warning: return the *smallest* cosine
%                                                   Use -query to obtain the largest
%                                   32=NBNN    -> Warning: return the *smallest* cosine
%                                                   Use -query to obtain the largest
%                    available in Mex-version only
%
% Returned values
%   idx         the vector index of the nearest neighbors
%   dis         the corresponding *square* distances
%
% Both v and q contains vectors stored in columns, so transpose them if needed
function [idx, dis] = my_nn (X, Q, k, distype, meanCG,sigma)


if ~exist('k'), k = 1; end
if ~exist('distype'), distype = 2; end
if ~exist('slicesize'), slicesize = 1000; end
assert (size (X, 1) == size (Q, 1));

% Settings for distance
switch distype
    case {2,'L2'}, mode = 'ascend';
    case {8,'L1'}, mode = 'ascend';
    case {32,'NBNN'}, mode = 'descend';
    case {34,'LOC'}, mode = 'ascend';
    case {64,'CKNK'}, mode = 'descend';
    case {16,'COS'}, mode = 'descend';
end



% Main loop
switch distype
    case {2,'L2'}
        % Compute half square norm
        X_nr = sum (X.^2) / 2;
        Q_nr = sum (Q.^2) / 2;
        sim = bsxfun (@plus, Q_nr', bsxfun (@minus, X_nr, Q'*X));
        
    case {16,'COS'}
        sim = Q' * X;
        
    case {8,'L1'}
        sim = Q' * X;
        
    case {32,'NBNN'}
        num_q = size(Q,2);
        num_x = size(X,2);
        sim=zeros(num_q,num_x);
        for i_q=1:num_q
            for j_x=1:num_x
                try
                    sim(i_q,j_x) = disNBNN(Q(i_q).CG,X(j_x).CG,meanCG);
                    %  fprintf( 'query = %d, data = %d, sim = %f, num_cg_q = %d, num_cg_d = %d\n',i_q,j_x,sim(i_q,j_x),...
                    %      size(Q(i_q).CG,2),size(X(j_x).CG,2));
                catch
                    sim(i_q,j_x) = 0;
                end
            end
        end
    case {34,'LOC'}
        num_q = size(Q,2);
        num_x = size(X,2);
        sim=zeros(num_q,num_x);
        for i_q=1:num_q
            for j_x=1:num_x
                try
                    sim(i_q,j_x) = disLOC(Q(i_q),X(j_x),sigma);
                    %  fprintf( 'query = %d, data = %d, sim = %f, num_cg_q = %d, num_cg_d = %d\n',i_q,j_x,sim(i_q,j_x),...
                    %      size(Q(i_q).CG,2),size(X(j_x).CG,2));
                catch
                    sim(i_q,j_x) = 1000;
                end
            end
        end
        
    case {64,'CKNK'}
        num_q = size(Q,2);
        num_x = size(X,2);
        sim=zeros(num_q,num_x);
        for i_q=1:num_q
            for j_x=1:num_x
                try
                    %sim(i_q,j_x) = disCKNK(Q(i_q).HK,X(j_x).HK,meanCG);
                    sim(i_q,j_x) = disCKNK_amap(Q(i_q),X(j_x),meanCG);
                    %  fprintf( 'query = %d, data = %d, sim = %f, num_cg_q = %d, num_cg_d = %d\n',i_q,j_x,sim(i_q,j_x),...
                     %     size(Q(i_q).CG,2),size(X(j_x).CG,2));
                catch
                    sim(i_q,j_x) = 0;
                end
            end
        end
        
    otherwise
        error ('Unknown distance type');
end


[dis, idx] = kmin_or_kmax (sim, k, mode);



% Post-processing for some metrics
switch distype
    case {2,'L2'}
        dis = dis * 2;
end


%------------------------------------------------
% Choose min or max depending on mode
function [dis,idx] = kmin_or_kmax (sim, k, mode)

if k == 1
    if mode(1) == 'a'
        [dis,idx] = min (sim, [], 2);
        dis = dis';
        idx = idx';
    else
        [dis,idx] = max (sim, [], 2);
        dis = dis';
        idx = idx';
    end
    
else  % k>1
    [dis, idx] = sort (sim, 2, mode);
    dis = dis (:, 1:k)';
    idx = idx (:, 1:k)';
end
