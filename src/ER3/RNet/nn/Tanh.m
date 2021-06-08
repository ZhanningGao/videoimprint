% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Tanh < Module
    properties
    end
    methods
        function obj = Tanh()
            obj = obj@Module();
        end
        function output = fprop(obj, input)
            exp_pos_x = exp(input);
            exp_neg_x = exp(-1*input);
            obj.output = (exp_pos_x - exp_neg_x)./(exp_pos_x + exp_neg_x);
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            exp_pos_x = exp(input);
            exp_neg_x = exp(-1*input);
            obj.grad_input = 2*grad_output ./(exp_pos_x + exp_neg_x);
            grad_input = obj.grad_input;
        end
    end
end