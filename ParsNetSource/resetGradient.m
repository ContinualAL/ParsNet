function FlexNetwork = resetGradient(FlexNetwork,hiddenlayer,outputlayer)
nHiddenLayer = FlexNetwork.nHiddenLayer;
if (hiddenlayer > nHiddenLayer && outputlayer > nHiddenLayer) || ...
    (hiddenlayer < 0 || outputlayer < 0)
    msg = 'Error: the deleted layer does not exist';
    error(msg)
end
if hiddenlayer == 0
    for iHiddenLayer = 1 : nHiddenLayer
        FlexNetwork.vW{iHiddenLayer}  = FlexNetwork.vW{iHiddenLayer}*0;
        FlexNetwork.dW{iHiddenLayer}  = FlexNetwork.dW{iHiddenLayer}*0;
    end
else
    for iHiddenLayer = 1 : hiddenlayer
        FlexNetwork.vW{iHiddenLayer}  = FlexNetwork.vW{iHiddenLayer}*0;
        FlexNetwork.dW{iHiddenLayer}  = FlexNetwork.dW{iHiddenLayer}*0;
    end
end
if outputlayer == 0
    for iHiddenLayer = 1:nHiddenLayer
        FlexNetwork.vWs{iHiddenLayer} = FlexNetwork.vWs{iHiddenLayer}*0;
        FlexNetwork.dWs{iHiddenLayer} = FlexNetwork.dWs{iHiddenLayer}*0;
    end
else
    for iHiddenLayer = outputlayer
        FlexNetwork.vWs{iHiddenLayer} = FlexNetwork.vWs{iHiddenLayer}*0;
        FlexNetwork.dWs{iHiddenLayer} = FlexNetwork.dWs{iHiddenLayer}*0;
    end
end
end