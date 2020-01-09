function [ParsNetwork,parameter] = ParsNet_init(nin,nClass)
parameter.nInput            = nin;
parameter.nClass            = nClass;
ParsNetwork                 = netconfig([nin nClass nClass]);
ParsNetwork.learningRate    = 0.01;%*(1 - lambda);            %  learning rate
ParsNetwork.learningRateDae = 0.001;%*lambda;
parameter.winninglayer      = 1;
parameter.t                 = 1;

%% initiate node evolving iterative parameters
layer                       = 1;     % number of layer
parameter.ev{1}.layer       = layer;
parameter.ev{1}.kp          = 0;
parameter.ev{1}.kl          = 0;
parameter.ev{1}.K           = nClass;
parameter.ev{1}.Kd          = 0;
parameter.ev{1}.node        = [];
parameter.ev{1}.BIAS2       = [];
parameter.ev{1}.VAR         = [];
parameter.ev{1}.miu_NS_old  = 0;
parameter.ev{1}.std_NS_old  = 0;
parameter.ev{1}.miu_NHS_old = 0;
parameter.ev{1}.std_NHS_old = 0;
parameter.ev{1}.miumin_NS   = [];
parameter.ev{1}.miumin_NHS  = [];
parameter.ev{1}.stdmin_NS   = [];
parameter.ev{1}.stdmin_NHS  = [];
parameter.ev{1}.grow_layer_count = 0;
parameter.ev{1}.noOfMissLabeling = 0;

%% initiate DAE parameter
parameter.dae{1}.timecreated    = 1;
parameter.dae{1}.K              = nClass;
parameter.dae{1}.Kg             = 1;
parameter.dae{1}.kk             = 0;
parameter.dae{1}.ky             = 0;
parameter.dae{1}.node           = [];
parameter.dae{1}.BIAS2          = [];
parameter.dae{1}.VAR            = [];
parameter.dae{1}.Loss           = [];
parameter.dae{1}.miu_x_tail_old = 0;
parameter.dae{1}.std_x_tail_old = 0;
parameter.dae{1}.miu_NS_old     = 0;
parameter.dae{1}.std_NS_old     = 0;
parameter.dae{1}.miu_NHS_old    = 0;
parameter.dae{1}.std_NHS_old    = 0;
parameter.dae{1}.miumin_NS      = [];
parameter.dae{1}.miumin_NHS     = [];
parameter.dae{1}.stdmin_NS      = [];
parameter.dae{1}.stdmin_NHS     = [];
parameter.dae{1}.grow_layer_count = 1;
parameter.dae{1}.mode           = 'sigmsigm';
parameter.dae{1}.minError       = inf;
parameter.dae{1}.maxError       = 0;

% initiate growing layer signal
parameter.ev{1}.grow_layer  = 0;
parameter.dae{1}.grow_layer = 0;
alpha                       = 0.005;
parameter.ev{1}.alpha       = alpha;
parameter.dae{1}.alpha      = alpha;
end

%% initiate neural network
function net = netconfig(layer)
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

net.size                 = layer;
net.nLayer               = numel(net.size);  %  Number of layer
net.nHiddenLayer         = net.nLayer - 2;   %  number of hidden layer
net.activation_function  = 'sigm';          %  Activation functions of hidden layers: 'sigm'.
net.learningRate         = 0.01;            %  learning rate
net.learningRateDae      = 0.001;
net.momentum             = 0.95;            %  Momentum
net.outputConnect        = 1;               %  1: connect all hidden layer output to output layer, otherwise: only the last hidden layer is connected to output
net.output               = 'softmax';       %  output layer 'softmax'
net.nInput               = layer(1);
net.nClass               = layer(end);
net.class_count          = zeros(1,layer(end));
net.class_prob           = zeros(1,layer(end));
net.incrementalClass     = 0;
net.weightImportanceCount= 0;

%% initiate weights and weight momentum for hidden layer
for iLayer = 2 : net.nLayer - 1
    net.W {iLayer - 1} = normrnd(0,sqrt(2/(net.size(iLayer-1)+1)),[net.size(iLayer),net.size(iLayer - 1)+1]);
    net.vW{iLayer - 1} = zeros(size(net.W{iLayer - 1}));
    net.dW{iLayer - 1} = zeros(size(net.W{iLayer - 1}));
    net.c{iLayer - 1}  = normrnd(0,sqrt(2/(net.size(iLayer-1)+1)),[net.size(iLayer - 1),1]);
    
    net.deltaLossW{iLayer - 1} = zeros(size(net.W{iLayer - 1}));
    net.sumDeltaLossW{iLayer - 1} = zeros(size(net.W{iLayer - 1}));
    net.deltaW{iLayer - 1} = zeros(size(net.W{iLayer - 1}));
    net.weightImportanceW{iLayer - 1} = zeros(size(net.W{iLayer - 1}));
end

%% initiate weights and weight momentum for output layer
for iLayer = 1 : net.nHiddenLayer
    net.Ws {iLayer}       = normrnd(0,sqrt(2/(size(net.W{iLayer},1)+1)),...
                        [net.size(end),net.size(iLayer+1)+1]);
    net.vWs{iLayer}       = zeros(size(net.Ws{iLayer}));
    net.dWs{iLayer}       = zeros(size(net.Ws{iLayer}));
    net.beta(iLayer)      = 1;
    net.betaOld(iLayer)   = 1;
    net.penalty(iLayer)   = 1;
    
    net.deltaLossWs{iLayer}        = zeros(size(net.Ws{iLayer}));
    net.sumDeltaLossWs{iLayer}     = zeros(size(net.Ws{iLayer}));
    net.deltaWs{iLayer}            = zeros(size(net.Ws{iLayer}));
    net.weightImportanceWs{iLayer} = zeros(size(net.Ws{iLayer}));
end
end