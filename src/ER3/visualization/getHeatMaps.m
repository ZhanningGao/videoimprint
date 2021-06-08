function [heatFlow, heatValue, heatFlow_WxW] = getHeatMaps(ql,probsMap,W,vData,colorMap)

sub_probs = getSubProbs(probsMap,W);

ql2D = reshape(permute(ql,[2,1,3]),[size(ql,1)*size(ql,2),size(ql,3)]);

num_f = size(ql,3);

heatFlow_WxW = zeros([W,W,num_f]);
heatValue = zeros(2,num_f);

for i_f = 1:num_f
    
    for i_k=1:size(sub_probs,3)
        
        heatFlow_WxW(:,:,i_f) = heatFlow_WxW(:,:,i_f) + ql2D(i_k,i_f)*double(sub_probs(:,:,i_k));
        
    end
    
    heatValue(1,i_f) = sum(sum(heatFlow_WxW(:,:,i_f)));
    heatValue(2,i_f) = max(max(heatFlow_WxW(:,:,i_f)));
    
end

heatValue = heatValue./repmat(sum(heatValue,2),[1,size(heatValue,2)]);

if size(vData,1)>1
    h = size(vData,1);
    w = size(vData,2);
    
    heatFlow = zeros([h,w*2,3,num_f],'uint8');
    
    for i_f = 1:num_f
        
        heatFlow(:,1:w,:,i_f) = vData(:,:,:,i_f);
%         heatFlow(:,w+1:end,1,i_f) = uint8(imresize(heatFlow_WxW(:,:,i_f),[h,w],'nearest'));
%         heatFlow(:,w+1:end,2,i_f) = uint8(imresize(heatFlow_WxW(:,:,i_f),[h,w],'nearest'));
%         heatFlow(:,w+1:end,3,i_f) = uint8(imresize(heatFlow_WxW(:,:,i_f),[h,w],'nearest'));
        heatFlow_img = (255*ind2rgb(uint8(imresize(heatFlow_WxW(:,:,i_f),[h,w])),colorMap));
        heatFlow(:,w+1:end,:,i_f) = uint8(0.4*double(vData(:,:,:,i_f)) + 0.6*heatFlow_img);
    end
else
    heatFlow = zeros([1,1,1,num_f],'uint8');
end