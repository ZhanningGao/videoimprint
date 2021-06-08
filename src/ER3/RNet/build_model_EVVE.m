
% construct model


memory = {};
model = Sequential();

proj = {};
for i = 1:nhops

    memory{i} = MemoryVec_spatial(config);

    memory{i}.nil_word = 0;%dict('nil');
    model.add(Duplicate());
    P1 = Parallel();
    P1.add(memory{i});
    if add_proj
        proj{i} = LinearNB(config.input_dim,config.input_dim);
        P1.add(proj{i});        
    else
        P1.add(Identity());
    end
    model.add(P1);
    model.add(AddTable());
    if add_nonlin
        model.add(ReLU());
    end
end

model.add(LinearNB(config.out_dim, config.output_size));

model.add(Softmax());

% share weights
if share_type == 1
    memory{1}.emb_query.share(model.modules{1});
    for i = 2:nhops
        memory{i}.emb_query.share(memory{i-1}.emb_out);
    end
 %   model.modules{end-1}.share(memory{end}.emb_out);
elseif share_type == 2
 %   model.modules{1}.share(memory{1}.emb_query);
    for i = 2:nhops
        memory{i}.emb_query.share(memory{1}.emb_query);
        memory{i}.emb_out.share(memory{1}.emb_out);
    end
end
if add_proj
    for i = 2:nhops
        proj{i}.share(proj{1});
    end
end
% cost
loss = CrossEntropyLossVec();
%loss = EuclideanLossTarget();
loss.size_average = false;
loss.do_softmax_brop = true;
model.modules{end}.skip_bprop = true;