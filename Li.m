function [WW]=Li(fea,K)
%如何求取Li     L=D-W
%因为是实例，所以K=5
%这里根据数据特征k=20为最好，因为有20种图


if ~exist('K','var')
    K = 5;
end
 
 [mFea,nSmp]=size(fea);
 label = litekmeans(fea,K,'Replicates',20);

DCol=[];
D=[];
options = [];
options.WeightMode = 'Binary';  
alpha=100;%前面会有假设，这里是实例，所以直接给出了一个值
%因为要存放矩阵，所以定义一个1*K的胞元
%A=cell（1，K）；
%A{:}={[0]};对所有胞元初始化
%http://www.ilovematlab.cn/thread-53246-1-1.html
A=cell(1,K);
WW=cell(1,K);
%L=cell(1,K);

%因为是复现文章，这里只给出每一个的图结构，没有给出组合图的L矩阵，如果需要只需改变循环。

%因为label并不是与矩阵fea的列时一一对应的，所以要提取每一类的信息，可通过下面程序实现
labelA=[];
labelA=[(1:mFea)',label];
for i=1:K
    for j=1:mFea
        if labelA[j,2]==i;  %找出label所对应的行，
    A{1,i}(j,:)=fea(j,:);    %提取这一行的所有信息
        end
    end
    WW{1,i}=constructW(A{1,i},options);
%     if alpha > 0
%         %最终求解释输入的矩阵，是转置后的。
%     WW{1,i} = alpha*WW{1,i};
%     DCol = full(sum(WW{1,i},2));
%     D = spdiags(DCol,0,nSmp,nSmp);
%     L{1,i} = D - WW{1,i};
%    else
%     L{1,i} = [];
%    end
end
    
    
  
    
   
    