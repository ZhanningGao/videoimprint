% probs to videos

W = 8;
E = 24;

% colorMap
load('colorMap.mat')

%probs
load('..\..\..\data\probs.mat');

%testQL
load('..\..\..\data\video_imprint\testQL.mat');


probs2 = (probs - 0);
probs2(probs2<0)=0;

probsMap = uint8((probs2-repmat(min(probs2),size(probs2,1),1))*255./repmat(max(probs2)-min(probs2),size(probs2,1),1));


probs3D = reshape(probsMap,[E,E,620]);

fps = 5;

%set the dataset path
videoPath = '..\..\..\data\video_examples';

mp4obj = VideoReader(fullfile(videoPath,'test.mp4'));
numFrames = get(mp4obj, 'NumberOfFrames');
framesRate = get(mp4obj, 'FrameRate');

vData = read(mp4obj);
vData = vData(:,:,:,1:3:end);
[h,w,c,num] = size(vData);
fprintf('Size = [%d x %d], frame = %d, Done\n', h,w,num);


[h,w,c,num] = size(vData);


% center crop
if h>w
    h_top = ceil((h/2)-(w/2) + 0.1);
    h_down = h_top+w-1;
    vData = vData(h_top:h_down,:,:,:);
else
    w_left = ceil((w/2)-(h/2)+0.1);
    w_right = w_left+h-1;
    vData = vData(:,w_left:w_right,:,:);
end
fprintf('Done\n');

probs_c = probs3D(:,:,47);


[heatFlow, heatValue] = getHeatMaps(testQL,probs_c,W,vData,colorMap);

playShotValue(heatFlow,heatValue,50,colorMap);