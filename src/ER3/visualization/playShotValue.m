function playShotValue(shot,heatValue,L,colorMap)

sz = size(shot);
sz(2) = sz(2)/2;

nFrames = size(shot,4);

%heatValue(1,:) = heatValue(1,:)/max(heatValue(1,:));
%heatValue(2,:) = heatValue(2,:)/max(heatValue(2,:));

heatValue(1,:) = (heatValue(1,:)-min(heatValue(1,:)))/(max(heatValue(1,:))-min(heatValue(1,:)));
heatValue(2,:) = (heatValue(2,:)-min(heatValue(2,:)))/(max(heatValue(2,:))-min(heatValue(2,:)));

for i=1:nFrames
    
    barsImg = showValueWithBars(heatValue(2,max(i-L+1,1):i),L,sz(1:2),colorMap,[0,1,0],[1,1,1]*0.95);
    
    img = [uint8(barsImg*255),shot(:,:,:,i)];
    
    imshow(img);title(num2str(heatValue(2,i)));drawnow;
    pause(1/200);
end

end