% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant 
% of patent rights can be found in the PATENTS file in the same directory.

classdef Weight < handle
    properties
        sz
        D
        grad
        old_w
    end
    methods
        function obj = Weight(sz)
            obj.sz = sz;
   %         obj.D = gpuArray(0.05 * randn(sz, 'single'));
            obj.D = gpuArray(1e-4 *  (mod( fix( 2^50  *  rand(sz, 'single') ) , 2000 )    -1000));
            for i=1:prod(sz) /2
                obj.D(ceil(rand*prod(sz)))=0;
            end
            obj.grad = gpuArray(zeros(sz, 'single'));
            obj.old_w= gpuArray(zeros(sz, 'single'));            
        end
        function update(obj, params)
            if isfield(params, 'max_grad_norm') && params.max_grad_norm > 0
                if norm(obj.grad) > params.max_grad_norm
                    obj.grad = obj.grad * params.max_grad_norm / norm(obj.grad);
                end
            end
            obj.old_w = params.momentum*obj.old_w  - params.lrate * obj.grad;
            obj.D = obj.D + obj.old_w;% - params.lrate * obj.grad;
            obj.grad(:) = 0;
        end
        function m = clone(obj)
            m = Weight(obj.sz);
            m.D = obj.D;
            m.grad = obj.grad;
            m.old_w= obj.old_w;
        end
    end
end