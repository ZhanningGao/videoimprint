% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef CrossEntropyLossVec < Loss
    properties
        do_softmax_brop = false;
        eps = 0.0000001;
        size_average = true;
    end
    methods
        function obj = CrossEntropyLossVec()
            obj = obj@Loss();
        end
        function cost = fprop(obj, input, target)
            %z = sub2ind(size(input), target, 1:length(target));
            t = bsxfun(@rdivide,target,sum(target,1));
            t(isnan(t))=1/size(target,1);
            cost = sum(-1*t(:).*log(input(:)));
            if obj.size_average
                cost = cost / size(input,2);
            end
        end
        function grad_input = bprop(obj, input, target)
            %z = sub2ind(size(input), target, 1:length(target));
            t = bsxfun(@rdivide,target,sum(target,1));

            t(isnan(t))=1/size(target,1);
            
            if obj.do_softmax_brop
                % better numberical stability
                grad_input = input - t;
            else
                grad_input = -t./(input + obj.eps);
            end
            if obj.size_average
                grad_input = grad_input / size(input,2);
            end
        end
        function error = get_error(obj, input, target)
            [~,y] = max(input,[],1);
            [~,t] = max(target,[],1);
            error = sum(y ~= t);
        end
    end    
end