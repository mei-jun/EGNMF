%元胞  之间的加法
function [W]=Cell_add(WW,R,K)
mm=size(WW{1,1},1);
W=zeros(mm,mm);
WR=cell(1,K);
for i=1:K
    WR{1,i}=WW{1,i}*R(i,1);
    W=W+WR{1,i};
end
