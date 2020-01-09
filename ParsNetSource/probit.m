%% calculate probit function
function p = probit(miu,std)
std = std + 0.0001;
p = (miu./(1 + pi.*(std.^2)./8).^0.5);
end
