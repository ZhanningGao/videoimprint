%extract average CG for each video as baseline2



%set the dataset path
videoSubDirPath = 'Y:\dataset\EVVE\CG-res5cx\CG_E24W8S4-100xWSS_L1norm_2';

subDir = dir(fullfile(videoSubDirPath,'*.mat'));

num_shot = 0;

num_dir = size(subDir,1);
id_v = 1;
cnnGCG = [];

active_node = 0;

%matlabpool local 12
for i=1:num_dir
    load(fullfile(videoSubDirPath,subDir(i).name));
    
    num_video = size(PIQL,2);
    for j_v = 1:num_video
        
        pi = (PIQL{j_v}.pi);
        
        Kmap = zeros(PIQL{j_v}.E,PIQL{j_v}.E);
        hk = pi;
        Wmap = sum(PIQL{j_v}.ql,3);
        
        Kmap(Wmap>8)=1;
        mask = zeros((PIQL{j_v}.W-1)*2+1);
        mask(PIQL{j_v}.W:end,PIQL{j_v}.W:end)=1;
        
        Kmap = circconv2(Kmap,mask,PIQL{j_v}.W-1,[PIQL{j_v}.E,PIQL{j_v}.E]);
        Kmap(Kmap>0)=1;
        
        Wmap = Wmap.^0;
        Wmap = Kmap.*Wmap;
        
        Wmap = reshape(Wmap,size(Kmap));
        
        repWmap = reshape(repmat(Wmap(:),[1,size(pi,3)]),size(pi));
        
        hk = hk.*repWmap;
        
        cgvMean = squeeze(sum(sum(hk,1),2));
        
        cgv = cgvMean;
        cgv = cgv./sum(cgv);
        
        cnnGCG(id_v).CG = cgv;
        
        active_node = active_node + sum(sum(Kmap));
        
        id_v = id_v + 1;
        
    end
    fprintf('=====SubDir %d/%d  Video done\n',i,num_dir);
end

