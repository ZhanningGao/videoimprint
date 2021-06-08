
batch_size = 124; 
output_size = 13;
input_size = 256;

nhops = 7;
nepochs = 20;


lrate_decay_step = 5; % reduce learning rate by half every 5 epochs

% use 10% of training data for validation
%EVVE_trID = randperm(size(EVVE_train,2));
load EVVE_trID.mat

train_range = EVVE_trID(1:min(floor(0.9 * length(EVVE_trID)),length(EVVE_trID) - batch_size));
val_range = EVVE_trID((min(floor(0.9 * length(EVVE_trID)),length(EVVE_trID) - batch_size)+1):length(EVVE_trID));

share_type = 2; % 1: adjecent, 2: layer-wise weight tying 
randomize_time = 0.1; % amount of noise injected into time index
add_proj = false; % add linear layer between internal states
add_nonlin = false; % add non-linearity to internal states

config = {};
config.init_lrate = 0.02;

config.momentum = 0.0;

config.max_grad_norm = 20;
config.input_dim = 256;
config.out_dim = 256;


config.timeW = 0.5;

config.sz = 576;
config.voc_sz = input_size;

config.output_size = output_size;
config.input_size = input_size;

config.bsz = batch_size;