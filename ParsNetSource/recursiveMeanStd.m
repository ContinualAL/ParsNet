%% calculate recursive mean and standard deviation
function [miu,std] = recursiveMeanStd(miu_old,std_old,x,k)
miu = miu_old + (x - miu_old)./k;
std = sqrt(std_old.^2 + miu_old.^2 - miu.^2 + ((x.^2 - std_old.^2 -...
    miu_old.^2)/(k)));
end