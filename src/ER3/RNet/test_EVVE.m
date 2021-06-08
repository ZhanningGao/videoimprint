
total_test_err = 0;
total_test_num = 0;

probs = zeros(size(memory{1}.probs,1),size(EVVE_test,2));
predict = zeros(size(model.output,1),size(EVVE_test,2));
predictLabel = zeros(1,size(EVVE_test,2));

for k = 1:floor(size(EVVE_test,2)/batch_size)
    batch = (1:batch_size) + (k-1) * batch_size;
    
    input = gpuArray(zeros(config.input_size,batch_size,'single'));
    
    target = gpuArray(EVVE_target_test(batch,:)');
    memory{1}.data(:) = 0;
    
    for b = 1:batch_size
        d = EVVE_test(batch(b)).CGset;
        offset(b) = max(0,size(d,2)-config.sz);
        d = d(:,1+offset(b):end);
        
        
        memory{1}.data(1:size(d,1),1:size(d,2),b) = d;
          
%        input(:,b) = mean(d,2);
        input(:,b) = yael_vecs_normalize(sum(d,2));
        input(isnan(input)) = 0;
    end
    for i = 2:nhops
        memory{i}.data = memory{1}.data;
    end
    
    out = model.fprop(input);
    for i = 1:nhops
        probs(:,batch) = probs(:,batch) + gather(memory{i}.probs);
    end
    
    predict(:,batch) = gather(out);
    [~,y] = max(out,[],1);
    predictLabel(batch) = gather(y);
    
    cost = loss.fprop(out, target);
    total_test_err = total_test_err + gather(loss.get_error(out, target));
    total_test_num = total_test_num + batch_size;
end
probs = probs/nhops;
test_error = total_test_err/total_test_num;
disp(['test error: ', num2str(test_error)]);

map = calmap(predict',EVVE_target_test)
