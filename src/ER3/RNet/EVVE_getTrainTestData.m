% produce the train and test data of EVVE

%load('..\..\..\data\video_imprint\EVVEallsetResNet.mat')
%load('..\..\..\data\video_imprint\EVVEinfo.mat')

EVVE_train = EVVEallset(find(EVVETrainID));
EVVE_test  = EVVEallset(find(EVVETestID));

EVVE_target_train = EVVELabel(find(EVVETrainID),:);
EVVE_target_test  = EVVELabel(find(EVVETestID),:);