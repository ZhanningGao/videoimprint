% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef AvePooling < Module
    properties
        R
    end
    methods
        function obj = AvePooling(R)
            obj = obj@Module();
            obj.R = R;
        end
        function output = fprop(obj, input)
            sz = size(input);
            input2D = padarray(reshape(input,[sqrt(sz(1)),sqrt(sz(1)),sz(2:end)]),[(obj.R-1)/2,(obj.R-1)/2,zeros(1,length(sz)-1)],'circular','both');
            mask = ones(obj.R,obj.R)/(obj.R)^2;
            output2D = convn(input2D,mask,'valid');
            obj.output = reshape(output2D,sz);
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            sz = size(grad_output);
            g_output2D = padarray(reshape(grad_output,[sqrt(sz(1)),sqrt(sz(1)),sz(2:end)]),[(obj.R-1)/2,(obj.R-1)/2,zeros(1,length(sz)-1)],'circular','both');
            mask = ones(obj.R,obj.R)/(obj.R)^2;
            g_input2D = convn(g_output2D,mask,'valid');
            obj.grad_input = reshape(g_input2D,sz);
            grad_input = obj.grad_input;
        end
    end
end