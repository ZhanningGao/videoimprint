% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant
% of patent rights can be found in the PATENTS file in the same directory.

classdef LinearNBdpout < Module
    properties
        
        train_step = true;
        
        in_dim;
        out_dim;
        dprate = 0.5;
        mask;
        weight;
        do_transpose = false;
    end
    methods
        function obj = LinearNBdpout(input_dim, output_dim, do_transpose, dprate)
            obj = obj@Module();
            obj.in_dim = input_dim;
            obj.out_dim = output_dim;
            
            
            obj.mask = gpuArray(ones(output_dim,1));
            
            if nargin == 4
                obj.do_transpose = do_transpose;
                obj.dprate = dprate;
            end
            if obj.do_transpose
                obj.weight = Weight([input_dim, output_dim]);
            else
                obj.weight = Weight([output_dim, input_dim]);
            end
        end
        function output = fprop(obj, input)
            if ndims(input) > 2
                sz = size(input);
                input = reshape(input, size(input,1),[]);
            end
            if obj.do_transpose
                obj.output = obj.weight.D' * input;
            else
                obj.output = obj.weight.D * input;
            end
            if exist('sz') == 1
                sz(1) = size(obj.output,1);
                obj.output = reshape(obj.output, sz);
            end
            if obj.train_step
                obj.mask(:) = 1;
                obj.mask(rand(obj.out_dim,1)<obj.dprate) = 0;
                maskMap = repmat(obj.mask,[1,size(input,2)]);
            else
                obj.mask(:) = 1 - obj.dprate;
                maskMap = repmat(obj.mask,[1,size(input,2)]);
            end
            
            obj.output = obj.output.*maskMap;
            output = obj.output;
        end
        function grad_input = bprop(obj, input, grad_output)
            maskMap = repmat(obj.mask,[1,size(input,2)]);
            grad_output = grad_output.*maskMap;
            if ndims(input) > 2
                sz = size(input);
                input = reshape(input, size(input,1),[]);
                grad_output = reshape(grad_output, size(grad_output,1),[]);
            end
            if obj.do_transpose
                obj.weight.grad = obj.weight.grad + input * grad_output';
                obj.grad_input = obj.weight.D * grad_output;
            else
                obj.weight.grad = obj.weight.grad + grad_output * input';
                obj.grad_input = obj.weight.D' * grad_output;
            end
            if exist('sz') == 1
                obj.grad_input = reshape(obj.grad_input, sz);
            end
            grad_input = obj.grad_input;
        end
        function update(obj, params)
            obj.weight.update(params);
        end
        function share(obj, m)
            obj.weight = m.weight;
        end
    end
end