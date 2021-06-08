
clear;
num_lab = 12;

%---add your own MatConvNet path ----------------------
MatConvNet_root = 'Y:\code\matconvnet-1.0-beta20';
addpath('Y:\code\matconvnet-1.0-beta20');
addpath('Y:\code\matconvnet-1.0-beta20\matlab');
run vl_setupnn;
%------------------------------------------------------


% assign gpus and load CNN models for each Job/Lab
gpus = [1,2,3,4];
spmd(num_lab)
    gpuDevice(gpus(ceil(labindex/(num_lab/length(gpus)))));
    net = dagnn.DagNN.loadobj(load(fullfile(MatConvNet_root,'model','imagenet-resnet-50-dag.mat'))) ;
    net.mode = 'test' ;
    net.conserveMemory = 1;
    net.vars(net.getVarIndex('res5cx')).precious = 1;
    net.move('gpu');
    averageImg = imresize(net.meta.normalization.averageImage,net.meta.normalization.imageSize(1:2)*2);
end


batchSize = 4;

% para of TCG. resize the last conv layer's output to S x S x channels
S = 4;

% downsample fps of video
fps = 5;

% set the dataset path of input video
videoSubDirPath = 'D:\v-zhaga\EVVE\videos';

% set the output CNN features' dir
OutputDir = 'Y:\dataset\EVVE\CNNfeature\ResNet_14x14';


subDir = dir(fullfile(videoSubDirPath));

num_dir = size(subDir,1);
num_j = 1;

for i=3:num_dir
    mkdir(fullfile(OutputDir,subDir(i).name));
    
    mp4file = dir(fullfile(videoSubDirPath,subDir(i).name,'*.mp4'));
    
    num_video = size(mp4file,1);
    for j_v = 1:num_lab:num_video
        
        spmd(num_lab)
            
            if j_v+labindex-1<=num_video
                fprintf('Reading video %s ......    ',mp4file(j_v+labindex-1).name);
                % decode the video
                mp4obj = VideoReader(fullfile(videoSubDirPath,subDir(i).name,mp4file(j_v+labindex-1).name));
                numFrames = get(mp4obj, 'NumberOfFrames');
                framesRate = get(mp4obj, 'FrameRate');
                
                vData = read(mp4obj);
                vData = vData(:,:,:,1:3:end);
                [h,w,c,num] = size(vData);
                fprintf('Size = [%d x %d], frame = %d, Done\n', h,w,num);
                
                
                fprintf('Croping video %s ......    ',mp4file(j_v+labindex-1).name);
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
                
                fprintf('Extracting video %s\n',mp4file(j_v+labindex-1).name);
                % tic
                res5cx = zeros(14,14,2048,num,'single');
                
                for j_batch = 1:floor(num/batchSize)
                    frame = single(vData(:,:,:,(j_batch-1)*batchSize+1:(j_batch)*batchSize));
                    frame = reshape(imresize(frame, net.meta.normalization.imageSize(1:2)*2),...
                        [net.meta.normalization.imageSize(1:2)*2,3,batchSize]);
                    frame = frame - repmat(averageImg,[1,1,1,batchSize]) ;
                    frame = gpuArray(frame);
                    % run the CNN
                    net.eval({'data', frame}) ;
                    
                    % obtain the CNN otuput
                    res5cx(:,:,:,(j_batch-1)*batchSize+1:(j_batch)*batchSize) = squeeze(gather(net.vars(net.getVarIndex('res5cx')).value));
                    
                end
                if mod(num,batchSize)~=0
                    frame = single(vData(:,:,:,floor(num/batchSize)*batchSize+1:end));
                    frame = reshape(imresize(frame, net.meta.normalization.imageSize(1:2)*2),...
                        [net.meta.normalization.imageSize(1:2)*2,3,num-floor(num/batchSize)*batchSize]);
                    frame = frame - repmat(averageImg,[1,1,1,num-floor(num/batchSize)*batchSize]) ;
                    frame = gpuArray(frame);
                    net.eval({'data', frame}) ;
                    
                    % obtain the CNN otuput
                    res5cx(:,:,:,floor(num/batchSize)*batchSize+1:end) = squeeze(gather(net.vars(net.getVarIndex('res5cx')).value));
                end
                %toc
                fprintf('Done\n');
                res5cx = single(res5cx);
                
                sumRes5cx = squeeze(sum(sum(res5cx,1),2));
                
                Res5 = res5cx;
                Res5 = Res5 ./ repmat(sum(Res5,3),[1,1,2048,1]);
                Res5 = imresize(Res5,[S,S]);
                
                Res5_2D = reshape(permute(Res5,[3,2,1,4]),[S*S*2048,size(Res5,4)]);

            else
                Res5_2D = 0;
            end
            
            fprintf('===============Video %d/%d done\n',j_v+labindex-1,num_video);
        end
        % save the CNN features
        for j=1:num_lab
            if j_v+j-1<=num_video
                cnn4v.res5cx = Res5_2D{j};
                cnn4v.sumRes5cx = sumRes5cx{j};
                cnn4v.name = mp4file(j_v+j-1).name;
                save(fullfile(OutputDir,subDir(i).name,[mp4file(j_v+j-1).name '.mat']),'cnn4v','-v7.3');
            end
        end
    end
    
end