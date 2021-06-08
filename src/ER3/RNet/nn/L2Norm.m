% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant
% of patent rights can be found in the PATENTS file in the same directory.

classdef L2Norm < Module
    properties
    end
    methods
        function obj = L2Norm()
            obj = obj@Module();
        end
        function output = fprop(obj, input)
            
            inputPlus = input + 10e-10;
            obj.output = inputPlus ./ repmat(sqrt(sum(inputPlus.*inputPlus,1)),[size(input,1),1]);
            
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            
            inputPlus = input + 10e-10;
            obj.grad_input = (grad_output - repmat(sum(obj.output.*grad_output,1),[size(grad_output,1),1]).*obj.output) ./ ...
                repmat(sqrt(sum(inputPlus.*inputPlus,1)),[size(inputPlus,1),1]);
            
            grad_input = obj.grad_input;
        end
    end
end