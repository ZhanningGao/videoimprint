
params = {};

params.lrate = config.init_lrate;
params.momentum = config.momentum;
params.max_grad_norm = config.max_grad_norm;

params.timeW = config.timeW;

TrainErr = gpuArray(zeros(3,20));




for ep = 1:nepochs
    probs_tr = zeros(576,size(EVVE_train,2));
    predict_tr = zeros(13,size(EVVE_train,2));
    predictLabel_tr = zeros(1,size(EVVE_train,2));
    tic
    if mod(ep, lrate_decay_step) == 0
        params.lrate = params.lrate * 0.5;
    end
    total_err = 0;
    total_cost = 0;
    total_num = 0;
    
    current_err = 0;
    current_cost = 0;
    current_num = 0;
    
    for k = 1:floor(length(train_range)/batch_size)
        batch = train_range(randi(length(train_range), batch_size,1));
        input = gpuArray(zeros(config.input_size,batch_size,'single'));
        
        target = gpuArray(EVVE_target_train(batch,:)');
        
        memory{1}.data(:) = 0;
        offset = zeros(1,batch_size,'single');
        for b = 1:batch_size
            d = gpuArray(EVVE_train(batch(b)).CGset);
            offset(b) = max(0,size(d,2)-config.sz);
            d = d(:,1+offset(b):end);
            
            
            memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
            
            
            % memory{1}.data(1,size(d,2)+1:size(memory{1}.data,2),b) = 1;
            %             num_nonzero = length(find(sum(d,1)>0));
            %             if num_nonzero==0
            %                 num_nonzero=1;
            %             end
            %             input(:,b) = sum(d,2)./num_nonzero;
            
            input(:,b) = yael_vecs_normalize(sum(d,2));
            input(isnan(input)) = 0;
        end
        for i = 2:nhops
            memory{i}.data = memory{1}.data;
        end
        
        out = model.fprop(input);
        
        for i = 1:nhops
            probs_tr(:,batch) = probs_tr(:,batch) + gather(memory{i}.probs);
        end
        
        predict_tr(:,batch) = gather(out);
        [~,y] = max(out,[],1);
        predictLabel_tr(batch) = gather(y);
        
        cost = loss.fprop(out, target);
        err = loss.get_error(out, target);
        
        current_cost = current_cost + cost;
        current_err = current_err + err;
        current_num = current_num + batch_size;
        
        total_cost = total_cost + cost;
        total_err = total_err + err;
        total_num = total_num + batch_size;
        grad = loss.bprop(out, target);
        model.bprop(input, grad);
        model.update(params);
        for i = 1:nhops
            %        memory{i}.emb_query.weight.D(:,1) = 0;
        end
        %
        %         if mod(k,5)==0
        %            current_err = current_err/current_num;
        %            current_cost = current_cost/current_num;
        %            disp(['iter-',num2str(k), ' | train error: ', num2str(current_err), ' ', num2str(current_cost)]);
        %            current_err = 0;
        %            current_cost = 0;
        %            current_num = 0;
        %         end
        
    end
    
    total_val_err = 0;
    total_val_cost = 0;
    total_val_num = 0;
    for k = 1:floor(length(val_range)/batch_size)
        % do validation
        batch = val_range((1:batch_size) + (k-1) * batch_size);
        input = gpuArray(zeros(config.input_size,batch_size,'single'));
        
        target = gpuArray(EVVE_target_train(batch,:)');
        
        memory{1}.data(:) = 0;
        for b = 1:batch_size
            d = gpuArray(EVVE_train(batch(b)).CGset);
            offset(b) = max(0,size(d,2)-config.sz);
            d = d(:,1+offset(b):end);
            
            
            memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
            
            %  memory{1}.data(1,size(d,2)+1:size(memory{1}.data,2),b) = 1;
            
            %input(:,b) = mean(d,2);
            %             num_nonzero = length(find(sum(d,1)>0));
            %             if num_nonzero==0
            %                 num_nonzero=1;
            %             end
            %             input(:,b) = sum(d,2)./num_nonzero;
            %             input(isnan(input)) = 0;
            input(:,b) = yael_vecs_normalize(sum(d,2));
            input(isnan(input)) = 0;
        end
        for i = 2:nhops
            memory{i}.data = memory{1}.data;
        end
        
        out = model.fprop(input);
        
        for i = 1:nhops
            probs_tr(:,batch) = probs_tr(:,batch) + gather(memory{i}.probs);
        end
        
        predict_tr(:,batch) = gather(out);
        [~,y] = max(out,[],1);
        predictLabel_tr(batch) = gather(y);
        
        
        total_val_cost = total_val_cost + gather(loss.fprop(out, target));
        total_val_err = total_val_err + gather(loss.get_error(out, target));
        total_val_num = total_val_num + batch_size;
    end
    train_error = total_err/total_num;
    train_cost = total_cost/total_num;
    val_error = total_val_err/total_val_num;
    
    TrainErr(1,ep) = train_cost;
    TrainErr(2,ep) = train_error;
    TrainErr(3,ep) = val_error;
    
    toc
    disp([num2str(ep), ' | train error: ', num2str(train_error), ' ', num2str(train_cost), ' | val error: ', num2str(val_error)]);
end

probs_tr = probs_tr/nhops;