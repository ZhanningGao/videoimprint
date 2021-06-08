
clear;

S = 4;
W = 8;

addpath('..\TCG');


gpus = [3,4];
spmd(12)
    gpuDevice(gpus(ceil(labindex/6)));
end

for E = [24]
    
    %set the dataset path of CNN feature
    videoSubDirPath = 'Y:\dataset\EVVE\CNNfeature\ResNet';
    
    % output TCG for each event
    OutputDir = ['Y:\dataset\EVVE\CG-res5cx\CGnew_E' num2str(E) 'W8S4-100xWSS_L2norm_2'];
    
    mkdir(OutputDir);
    subDir = dir(fullfile(videoSubDirPath));
    
    num_dir = size(subDir,1);
    num_j = 1;
    
    for i=3:num_dir
        
        matfile = dir(fullfile(videoSubDirPath,subDir(i).name,'*.mat'));
        
        num_mat = size(matfile,1);
        PIQL = [];
        
        for j_v = 1:12:num_mat
            
            spmd(12)
                if j_v+labindex-1<=num_mat
                    
                    fprintf('Reading matfile %s ......    ',matfile(j_v+labindex-1).name);
                    h = load(fullfile(videoSubDirPath,subDir(i).name,matfile(j_v+labindex-1).name));
                    
                    Res5_2D = h.cnn4v.res5cx;
                    
                    fprintf('Compute pi and ql of %s ......       ',matfile(j_v+labindex-1).name);
                    
                    [pi, ql, amap, hk] = getQLviaPI_gpu(Res5_2D, 0, E, W, S, 1, 1000, 4);
                    
                    fprintf('Done\n');
                    
                    PIQL_c.W = W;
                    PIQL_c.E = E;
                    PIQL_c.S = S;
                    
                    PIQL_c.frames = size(h.cnn4v.res5cx,4);
                    PIQL_c.pi = pi;
                    
                    PIQL_c.ql = ql;
                    
                    PIQL_c.sumRes5cx = sum(h.cnn4v.sumRes5cx,2);
                    
                else
                    PIQL_c = [];
                end
                
                fprintf('===============Video %d/%d done\n',j_v+labindex-1,num_mat);
            end
            for j=1:12
                if j_v+j-1<=num_mat
                    PIQL{j_v+j-1} = PIQL_c{j};
                    PIQL{j_v+j-1}.name = matfile(j_v+j-1).name(1:end-8);
                end
            end
        end
        
        fprintf('Save pi and ql of %s ......       ',subDir(i).name);
        save(fullfile(OutputDir,[subDir(i).name '.mat']),'PIQL','-v7.3');
        fprintf('Done\n');
        
    end
    
end
