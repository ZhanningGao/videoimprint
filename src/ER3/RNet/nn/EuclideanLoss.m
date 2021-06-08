% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef EuclideanLoss < Loss
    properties
        eps = 0.0000001;
    end
    methods
        function obj = EuclideanLoss()
            obj = obj@Loss();
        end
        function cost = fprop(obj, input)
            
            cost = mean(sum((input{1}-input{2}).^2,1),2)/2;
            
        end
        function grad_input = bprop(obj, input)
            
            grad_input{1} = (input{1}-input{2})/size(input{1},2);
            grad_input{2} = (input{2}-input{1})/size(input{1},2);
            
        end
    end    
end