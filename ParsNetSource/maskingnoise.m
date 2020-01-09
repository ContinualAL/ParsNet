function in = maskingnoise(in,nin,noiseIntensity)
%% input masking
if nin > 1
    if noiseIntensity > 0
        nMask = max(round(noiseIntensity*nin),1);
        mask_gen = randperm(nin,nMask);
        in(1,mask_gen) = 0;
    end
else
    mask_gen = rand(size(in(1,:))) > 0.3;
    in = in*mask_gen;
end
end