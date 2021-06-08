% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef ZeroInf < Module
    properties
    end
    methods
        function obj = ZeroInf()
            obj = obj@Module();
        end
        function output = fprop(obj, input)
            obj.output = input;
            obj.output(input==0) = -10;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            obj.grad_input = grad_output;
            %obj.grad_input(input==0) = -100;
            grad_input = obj.grad_input;
        end
    end
end