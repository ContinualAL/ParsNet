function [Data,chunk_size,M,nFolds] = dataprep(data,I,chunkSize)
[nData,nCol] = size(data);
M = nCol - I;
l = 0;
nFolds       = round(length(data)/chunkSize);                 % number of data chunk
chunk_size   = round(nData/nFolds);
round_nFolds = floor(nData/chunk_size);
Data = {};
if round_nFolds == nFolds
    if nFolds   == 1
        Data{1} = data;
    else
        for i=1:nFolds
            l = l+1;
            if i < nFolds
                Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            elseif i == nFolds
                Data1 = data(((i-1)*chunk_size+1):end,:);
            end
            Data{l} = Data1;
        end
    end
else
    if nFolds == 1
        Data{1} = data;
    else
        for i=1:nFolds-1
            l=l+1;
            Data1 = data(((i-1)*chunk_size+1):i*chunk_size,:);
            Data{l} = Data1;
        end
        i = nFolds;
        Data{nFolds} = data(((i-1)*chunk_size+1):end,:);
    end
end
end