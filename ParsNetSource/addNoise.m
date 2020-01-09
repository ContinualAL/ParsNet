function [input,target] = addNoise(data,label,kind,level)
[nData,nInput] = size(data);
if kind == 1 % masking noise
    for iData = 1:nData
        noisyData(iData,:) = maskingnoise(data(iData,:),nInput,level);
    end
elseif kind == 2 % gaussian noise
    noisyData = data + normrnd(0,level);
end
input = [data;noisyData];
target = [label;label];
end

