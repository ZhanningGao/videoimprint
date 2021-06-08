% Copyright (c) 2015-present, Facebook, Inc.
% All rights reserved.
%
% This source code is licensed under the BSD-style license found in the
% LICENSE file in the root directory of this source tree. An additional grant
% of patent rights can be found in the PATENTS file in the same directory.

classdef BNorm < Module
    properties
        mode;
        dim;
        gamma;
        bias;
        
        Mu;
        Var;
        
        Mu_sum;
        Var_sum;

        batchNum;
        
        epsilon;
    end
    methods
        function obj = BNorm(dim,mode)
            obj = obj@Module();
            obj.dim  = dim;
            obj.mode = mode;% test = 1;
            
            obj.gamma = Weight([dim, 1]);
            obj.bias = Weight([dim, 1]);
            
       %     obj.gamma.D(:) = 1;
       %     obj.bias.D(:) = 0;
            
            obj.Mu_sum = gpuArray(zeros(dim,1,'single'));
            obj.Var_sum = gpuArray(zeros(dim,1,'single'));
            
            obj.Mu = gpuArray(zeros(dim,1,'single'));
            obj.Var = gpuArray(zeros(dim,1,'single'));
            
            obj.batchNum = 0;
            
            obj.epsilon = 1e-4;
            
        end
        function output = fprop(obj, input)
            
            
            if obj.mode == 1
                norm_factor = obj.gamma.D./sqrt(obj.Var + obj.epsilon);
                output = bsxfun(@times, input, norm_factor);
                output = bsxfun(@plus, output, obj.bias.D -  norm_factor .* obj.Mu);
                
                obj.output = output;
            else
                mu = mean(input,2);
                x_mu = bsxfun(@minus,input,mu);
                sigma2 = mean(x_mu.^2,2);
                norm_factor = obj.gamma.D./sqrt(sigma2+obj.epsilon);
                output = bsxfun(@times, input, norm_factor);
                output = bsxfun(@plus, output, obj.bias.D -  norm_factor .* mu);
                
                obj.Mu_sum = obj.Mu_sum + mu;
                obj.Var_sum = obj.Var_sum + sigma2;
                
                obj.batchNum = obj.batchNum + 1;
                
                obj.output = output;
                
            end;
        end
        function grad_input = bprop(obj, input, grad_output)
            
            m = size(input,2);
            
            mu = mean(input,2);
            x_mu = bsxfun(@minus, input, mu);
            sigma2 = mean(x_mu.^2,2);
            
            xhat = bsxfun(@times,x_mu,1./sqrt(sigma2+obj.epsilon));
            
            d_xhat = bsxfun(@times, grad_output, obj.gamma.D);
            
            inv_sqrt_sigma = 1 ./ sqrt(sigma2 + obj.epsilon);
            d_sigma2 = -0.5 * sum(d_xhat .* x_mu,2) .* inv_sqrt_sigma.^3;
            d_mu = bsxfun(@times, d_xhat, inv_sqrt_sigma);
            d_mu = -1 * sum(d_mu,2) -2 .* d_sigma2 .* mean(x_mu,2);
            obj.gamma.grad = mean(grad_output .* xhat,2);
            obj.bias.grad = mean(grad_output,2);
            di1 = bsxfun(@times,d_xhat,inv_sqrt_sigma);
            di2 = 2/m * bsxfun(@times, d_sigma2,x_mu);
            obj.grad_input = di1 + di2 + 1/m * repmat(d_mu,1,m);

            grad_input = obj.grad_input;
        end
        function update(obj, params)
            obj.gamma.update(params);
            obj.bias.update(params);
        end
        function share(obj, m)
            obj.gamma = m.gamma;
            obj.bias = m.bias;
        end
    end
end