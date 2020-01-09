%% feedforward, to calculate network variance/bias
function [sigma,sigma2,hl_output,hl_output2,yl,yl2] = ...
    netfeedforwardf(net,gmm,winninglayer)
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

n = net.nLayer;
nClass = size(net.as{1},2);

%% feedforward from input layer through all the hidden layer
Ey = [];
for gmmk = 1:gmm.k
    if gmm.weight(gmmk) ~= 0
        py = probit(gmm.TrackerC(gmmk,:),...
            sqrt(gmm.TrackerS(gmmk,:)));
        py = [1 py];
        Ey(:,gmmk) = gmm.weight(gmmk)*...
            sigmf(net.W{1}*py',[1,0]); %  input expectation Equation
    end
end
if size(Ey,2) > 1
    Ey = sum(Ey,2);     % hidden node significance of the first hidden layer
end
Ey2 = Ey.^2;

activity{2}  = [1;Ey]';                % the first activity is the input itself
a2{2} = [1;Ey2]';               % the first activity is the input itself

for i = 3 : n-1
    activity{i}  = sigmf(activity{i - 1} * net.W{i - 1}',[1,0]);
    activity{i}  = [1 activity{i}];
    a2{i} = sigmf(a2{i - 1} * net.W{i - 1}',[1,0]);
    a2{i} = [1 a2{i}];
end

%% propagate to the output layer
sigma  = zeros(1,nClass);
sigma2 = zeros(1,nClass);
net.beta = net.beta./sum(net.beta);    % normalize voting weight
for i = 1 : net.nHiddenLayer
    if net.beta(i) ~= 0
        switch net.output
            case 'sigm'
                as{i} = sigmf(activity{i + 1} * net.Ws{i}',[1,0]);
                as2{i} = sigmf(a2{i + 1} * net.Ws{i}',[1,0]);
            case 'linear'
                as{i} = activity{i + 1} * net.Ws{i}';
                as2{i} = a2{i + 1} * net.Ws{i}';
            case 'softmax'
                as{i} = activity{i + 1} * net.Ws{i}';
                as{i} = exp(as{i} - max(as{i},[],2));
                as{i} = as{i}./sum(as{i}, 2);
                as2{i} = a2{i + 1} * net.Ws{i}';
                as2{i} = exp(as2{i} - max(as2{i},[],2));
                as2{i} = as2{i}./sum(as2{i}, 2);
        end
        if i == winninglayer
            hl_output  = activity{i + 1}(:,2:end);
            hl_output2 = a2{i + 1}(:,2:end);
            yl  = as{i};
            yl2 = as2{i};
        end
        sigma  = sigma  + as{i}*net.beta(i);%.*ev{i}.class_prob;
        sigma2 = sigma2 + as2{i}*net.beta(i);%.*ev{i}.class_prob;
    end
end
end