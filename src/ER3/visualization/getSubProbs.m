function sub_porbs = getSubProbs(probs_c,W)

probs_c_pad = padarray(probs_c,[W-1,W-1],'circular','post');

for k_i = 1:size(probs_c,1)
    for k_j = 1:size(probs_c,2)
        
       sub_porbs(:,:,(k_i-1)*size(probs_c,2)+k_j) = probs_c_pad(k_i:k_i+W-1,k_j:k_j+W-1); 
        
    end
end