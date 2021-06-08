function newqMat = DoN(fMat,qMat,idx,N1,N2)

pos_qMat = qMat;

for i=2:N1+1
    
    pos_qMat = pos_qMat + fMat(:,idx(i,:));
    
end

pos_qMat = pos_qMat/(N1+1);

if N2>N1
    neg_qMat = fMat(:,idx(N1+2,:));
    
    for i=N1+3:N2+N1+2
        
        neg_qMat = neg_qMat + fMat(:,idx(i,:));
        
    end
    
    neg_qMat = neg_qMat/(N2);
    
else
    
    neg_qMat = zeros(size(qMat));
end

newqMat = pos_qMat - neg_qMat;
