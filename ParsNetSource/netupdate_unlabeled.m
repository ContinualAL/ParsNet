%% update the weight
function net = netupdate_unlabeled(net,winninglayer,Wstar,WsStar,trueLabel)
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

% nHiddenLayer = nn.nHiddenLayer;
% for iHiddenLayer = winninglayer
initialW = net.W{winninglayer};
initialWs = net.Ws{winninglayer};
for iHiddenLayer = winninglayer%:nHiddenLayer
    if iHiddenLayer == winninglayer
        gradW = net.dW{winninglayer};
        gradW = net.learningRate * gradW;
        regularizerW = net.learningRate*net.weightImportanceW{winninglayer}.*(net.W{winninglayer} - Wstar{winninglayer});
        if(net.momentum > 0)
            net.vW{winninglayer} = net.momentum*net.vW{winninglayer} + gradW;
            gradW = net.vW{winninglayer};
        end
        
        %% update
        net.W{winninglayer} = net.W{winninglayer} - gradW - net.regStr*regularizerW;
        
        %% reset value for parameter importance
        if trueLabel == 1
            afterW = net.W{winninglayer};
            net.deltaW{winninglayer} = afterW - initialW;
            net.deltaLossW{winninglayer} = net.deltaW{winninglayer}.*gradW;
            net.sumDeltaLossW{winninglayer} = net.sumDeltaLossW{winninglayer} + ...
                net.deltaLossW{winninglayer};
        end
    end
    
    if net.beta(iHiddenLayer) ~= 0
        gradWs = net.dWs{iHiddenLayer};
        gradWs = net.learningRate*gradWs;
        regularizerWs = net.learningRate*net.weightImportanceWs{winninglayer}.*(net.Ws{winninglayer} - WsStar{winninglayer});
        if(net.momentum > 0)
            net.vWs{iHiddenLayer} = net.momentum*net.vWs{iHiddenLayer} + gradWs;
            gradWs = net.vWs{iHiddenLayer};
        end
        
        %% update
        net.Ws{iHiddenLayer} = net.Ws{iHiddenLayer} - gradWs - net.regStr*regularizerWs;
        
        %% reset value for parameter importance
        if trueLabel == 1
            afterWs = net.Ws{winninglayer};
            net.deltaWs{winninglayer} = afterWs - initialWs;
            net.deltaLossWs{winninglayer} = net.deltaWs{winninglayer}.*gradWs;
            net.sumDeltaLossWs{winninglayer} = net.sumDeltaLossWs{winninglayer} + ...
                net.deltaLossWs{winninglayer};
        end
    end
end
end