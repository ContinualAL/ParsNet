%% feedforward operation
function net = netFeedForward(net, input, label)
% This function is developed with modification from Rasmus Berg Palm
% Copyright (c) 2012, Rasmus Berg Palm (rasmusbergpalm@gmail.com)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met:
% 
% Redistributions of source code must retain the above copyright notice, 
% this list of conditions and the following disclaimer.
% 
% Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation 
% and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
% OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
% IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

nLayer = net.nLayer;
nData = size(input,1);
input = [ones(nData,1) input];          % by adding 1 to the first coulomn, it means the first coulomn of W is bias
net.a{1} = input;                 % the first activity is the input itself

%% feedforward from input layer through all the hidden layer
for iLayer = 2 : nLayer-1
    switch net.activation_function
        case 'sigm'
            net.a{iLayer} = sigmf(net.a{iLayer - 1} *...
                net.W{iLayer - 1}',[1,0]);
        case 'relu'
            net.a{iLayer} = max(net.a{iLayer - 1} *...
                net.W{iLayer - 1}',0);
    end
    net.a{iLayer} = [ones(nData,1) net.a{iLayer}];
end

%% propagate to the output layer
for iHiddenLayer = 1 : net.nHiddenLayer
    if net.beta(iHiddenLayer) ~= 0
        switch net.output
            case 'sigm'
                net.as{iHiddenLayer} = sigmf(net.a{iHiddenLayer + 1} *...
                    net.Ws{iHiddenLayer}',[1,0]);
            case 'linear'
                net.as{iHiddenLayer} = net.a{iHiddenLayer + 1} *...
                    net.Ws{iHiddenLayer}';
            case 'softmax'
                net.as{iHiddenLayer} = net.a{iHiddenLayer + 1} *...
                    net.Ws{iHiddenLayer}';
                net.as{iHiddenLayer} = exp(net.as{iHiddenLayer} -...
                    max(net.as{iHiddenLayer},[],2));
                net.as{iHiddenLayer} = net.as{iHiddenLayer}./...
                    sum(net.as{iHiddenLayer}, 2);
        end
        
        %% calculate error
        if isempty(label) == 0
            net.e{iHiddenLayer} = label - net.as{iHiddenLayer};
            
            %% calculate loss function
            switch net.output
                case {'sigm', 'linear'}
                    net.L(iHiddenLayer) = 1/2 *...
                        sum(sum(net.e{iHiddenLayer} .^ 2)) / nData;
                case 'softmax'
                    net.L(iHiddenLayer) = -sum(sum(label .* log(net.as{iHiddenLayer}))) / nData;
            end
        end
    end
end
end