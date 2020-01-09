function [Nc,Ni] = gmmUpdateLabel(gmmwinner,targetLabeledData,Nc,Ni)
%for calculating Supy
Ni(gmmwinner)   = Ni(gmmwinner) + 1;
Nc(gmmwinner,:) = Nc(gmmwinner,:) + targetLabeledData;
end