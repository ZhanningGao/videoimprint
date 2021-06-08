%evaluation the retrieval performance given the image representaions
%-----------by Zhanning Gao 2015/12/31----All for MT&G
%INPUT---------------------------------------------------------------------
% gnd - > groundtruth of dataset
%         format: Event(i)--- name
%                         |- num_query
%                         |- num_postive
%                         |- gnd ---query
%                                 |- junk    [1 x num_query]
%                                 |- ok      [1 x num_postive]
% f   - > database vectors for evaluation
%         format: f(i) --- name
%                       |- fc6               [4096 x 1]
%                       |... waiting for other repersentations
% featureType - > 'fc6'  ... waiting for other repersentations
% query-> query ID and class ID for the database
%         format: query --- [num_allquery x 2]
%                 query(i,:) --- [eventID queryID]
% disType - > 'fc6'  ... waiting for other repersentations
%OUTPUT--------------------------------------------------------------------
% mAP - > mAP for each event class           [num_event x 1]
% averagemAP - > mean(mAP)

function [averagemAP, mAP] = evalEVVE(f,query,gnd,meanCGeach,featureType,disType, PCAw_GCG)

if ~exist('featureType'), featureType = 'fc6'; end
if ~exist('disType'), disType = 'COS'; end

num_event = size(gnd,2);
num_f = size(f,2);
%dim = size(getfield(f(1),featureType),1);


%get database vectors matrix  fMat = [dim x num_f]
if ~strcmp(featureType,'CG_each')
    dim = size(getfield(f(1),featureType),1);
    fMat = zeros(dim,num_f);
end

for i=1:num_f
    switch featureType
        case {'Res5'}
            fMat(:,i) = yael_vecs_normalize(getfield(f(i),featureType));
        case {'CG'}
            fMat(:,i) = sign(getfield(f(i),featureType)).*abs(getfield(f(i),featureType)).^0.2;
    end
end

if ~strcmp(featureType,'CG_each')%&&~strcmp(featureType,'CKN')
    
    fMat = yael_vecs_normalize(fMat);
    fMat(isnan(fMat))=0;



    fMat = PCAw_GCG*bsxfun(@minus,fMat,meanCGeach);

    fMat = yael_vecs_normalize(fMat(1:1024,:));

    fMat(isnan(fMat))=0;

end

for i_e = 1:num_event
    %get query vectors matrix for event i_e    qMat = [dim x gnd(i_e).num_query]
    queryID = query(query(:,1)==i_e,2);
    if ~strcmp(featureType,'CG_each')
        qMat = fMat(:,queryID);
    else
        qMat = fMat(queryID);
    end
    
    [idx, dis] = my_nn (fMat, qMat, 2995, disType, meanCGeach, 0);
    
    qqMat = DoN(fMat,qMat,idx,10,2000);

    [idx_re, dis] = my_nn (fMat, qqMat, 2995, disType, meanCGeach, 0);
    
    [mAP(i_e), aps] = compute_map(idx,gnd(i_e).gnd);
    [mAP_re(i_e), aps] = compute_map(idx_re,gnd(i_e).gnd);
    fprintf('%d event mAP = %0.1f, with DoN = %0.1f\n',i_e,mAP(i_e)*100,mAP_re(i_e)*100);
end

averagemAP = mean(mAP);
averagemAP_re = mean(mAP_re);
fprintf('------------------------------------\n');
fprintf(' Average mAP = %0.1f, with DoN = %0.1f\n',averagemAP*100,averagemAP_re*100);


