% CC BY-NC-SA 4.0 License
% 
% Copyright (c) 2019 Mahardhika Pratama Andri Ashfahani Mohamad Abdul Hady

%% calculate backpropagation

function nn = netbackpropagation(nn,winninglayer)
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

nHiddenLayer = nn.nHiddenLayer;
% for iHiddenLayer = winninglayer
for iHiddenLayer = winninglayer%:nHiddenLayer
    nLayer = iHiddenLayer + 2;
    if iHiddenLayer == winninglayer
        %% calculate the gradient of the winning layer
        switch nn.output
            case 'sigm'
                activityDerivative{nLayer} = - nn.e{iHiddenLayer} .*...
                    (nn.as{iHiddenLayer} .* (1 - nn.as{iHiddenLayer}));
            case {'softmax','linear'}
                activityDerivative{nLayer} = - nn.e{iHiddenLayer};          % dL/dy
        end
        
        for i = (nLayer - 1) : -1 : (nLayer - 2)
            switch nn.activation_function
                case 'sigm'
                    activationDerivative = nn.a{i} .* (1 - nn.a{i}); % contains b
                case 'relu'
                    activationDerivative = zeros(1,length(nn.a{i}));
                    activationDerivative(nn.a{i}>0) = 0.1;
                    activationDerivative(nn.a{i}<=0) = 0.001;
            end
            
            if i+1 == nLayer
                activityDerivative{i} = (activityDerivative{i + 1} * nn.Ws{iHiddenLayer}) .* activationDerivative;
            else
                activityDerivative{i} = (activityDerivative{i + 1}(:,2:end) * nn.W{i})    .* activationDerivative;
            end
        end
        
        for i = (nLayer - 2) : (nLayer - 1)
            if i + 1 == nLayer
                nn.dWs{iHiddenLayer} = (activityDerivative{i + 1}' * nn.a{i});
            else
                nn.dW{i} = (activityDerivative{i + 1}(:,2:end)' * nn.a{i});
            end
        end
    else
        %% calculate the gradient of the softmax layer of other layer
        if nn.beta(iHiddenLayer) ~= 0
            switch nn.output
                case 'sigm'
                    ds = - nn.e{iHiddenLayer} .*...
                        (nn.as{iHiddenLayer} .* (1 - nn.as{iHiddenLayer}));
                case {'softmax','linear'}
                    ds = - nn.e{iHiddenLayer};          % dL/dy
            end
            i = (nLayer - 1);
            nn.dWs{iHiddenLayer} = (ds' * nn.a{i});
        end
    end
end
end