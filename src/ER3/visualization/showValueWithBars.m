%show heatValue

function BarsImg = showValueWithBars(heatValue,L,ImgSz,colorMap,barColor,backColor)

BarsImg = ones([160,(L+L/2),3]);
BarsImg(:,:,1) = backColor(1);
BarsImg(:,:,2) = backColor(2);
BarsImg(:,:,3) = backColor(3);

ColorScale = 230;%max = 255

i_h = length(heatValue);
for i=L:-1:L-length(heatValue)+1
    
    BarsImg(1:30,i,:) = repmat(reshape(colorMap(floor(heatValue(i_h)*ColorScale+1),:),[1,1,3]),[30,1,1]);
    
    batLocation = floor(heatValue(i_h)*99+1);
    BarsImg(160-batLocation+1:end,i,:) = repmat(reshape(barColor,[1,1,3]),[batLocation,1,1]);
    
    i_h = i_h - 1;
    
end

BarsImg = imresize(BarsImg,ImgSz,'nearest');