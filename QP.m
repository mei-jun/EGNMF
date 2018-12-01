%二次规划
function[R,fval]=QP(L,V,K)
%%WW  表示图像的邻接矩阵（这里是胞元）
%L   表示矩阵的L=D-WW,拉普拉斯矩阵
%V   矩阵分解中的X=UV’  因为二次规划涉及到Tr(V'LV)
% K  表示流行空间中潜藏的可能图的个数

%R   返回最小值是每个ri的取值    矩阵为K*1
%fval   目标函数值（最小的时候)
%  exitflag是描述搜索是否收敛；output是返回包含优化信息的结构。Lambda是返回解x入包含拉格朗日乘子的参数。
H=[];
f=[];
A=[];
b=[];
lb=[];
R=[];
fval=0;

H=diag(ones(1,K));

for i=1:K
    f(1,i)= sum(sum((V'*L{1,i}).*V'));
end
f=f';
A=ones(1,K);
A=[A;-A];
b=[1;-1];
lb=zeros(K,1);
%[R,fval,exitflag,output,lambda]=quadprog(H,f,A,b,[],[],lb);
[R,fval]=quadprog(H,f,A,b,[],[],lb);

    



