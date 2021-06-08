% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef EuclideanLossTarget < Loss
    properties
        eps = 0.0000001;
    end
    methods
        function obj = EuclideanLossTarget()
            obj = obj@Loss();
        end
        function cost = fprop(obj, input, target)
            
            cost = mean(sum((input-target).^2,1),2)/2;
            
        end
        function grad_input = bprop(obj, input, target)
            
            grad_input = (input-target);%/size(input,2);
            
        end
        function error = get_error(obj, input, target)
            [~,y] = max(input,[],1);
            [~,t] = max(target,[],1);
            error = sum(y ~= t);
        end
    end    
end