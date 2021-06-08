
gpuDevice(1);
rng('shuffle')
addpath nn;
addpath memory;

addpath ..\..\utils

% train and test
EVVE_getTrainTestData;

config_EVVE;

for nhops=[1,3,5,7]
    
    build_model_EVVE;
    train_EVVE;
    test_EVVE;
    
    save(['TrainErrHopsNew-' num2str(nhops) '.mat'], 'TrainErr');
    
end