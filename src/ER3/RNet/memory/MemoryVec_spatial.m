% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef MemoryVec_spatial < Memory
    properties
    end
    methods
        function obj = MemoryVec_spatial(config)
            obj = obj@Memory(config);
            obj.voc_sz = config.input_size;
            obj.data = gpuArray(zeros(obj.voc_sz, obj.sz, config.bsz, 'single'));
        end
        function initQueryModule(obj)
            obj.emb_query = LinearNB(obj.voc_sz ,obj.in_dim);            
            %S = Sequential();
            %S.add(obj.emb_query);
            %S.add(Sum(2));
            P = Parallel();
            P.add(obj.emb_query);
            P.add(Identity());
            obj.mod_query = Sequential();
            obj.mod_query.add(P);            
            obj.mod_query.add(MatVecProd(true)); 
            
            
          %  obj.mod_query.add(ZeroInf()); 
            
            obj.mod_query.add(Softmax());
            obj.mod_query.add(AvePooling(3));

            
        end
        function initOutputModule(obj)
            obj.emb_out = LinearNB(obj.voc_sz ,obj.out_dim);  
           % obj.emb_out = Identity();              
            %S = Sequential();
            %S.add(obj.emb_out);
            %S.add(Sum(2));
            P = Parallel();
            P.add(obj.emb_out);
            P.add(Identity());
            obj.mod_out = Sequential();
            obj.mod_out.add(P);            
            obj.mod_out.add(MatVecProd(false));            
        end
    end
end