function [WW]=Li(fea,K)
%�����ȡLi     L=D-W
%��Ϊ��ʵ��������K=5
%���������������k=20Ϊ��ã���Ϊ��20��ͼ


if ~exist('K','var')
    K = 5;
end
 
 [mFea,nSmp]=size(fea);
 label = litekmeans(fea,K,'Replicates',20);

DCol=[];
D=[];
options = [];
options.WeightMode = 'Binary';  
alpha=100;%ǰ����м��裬������ʵ��������ֱ�Ӹ�����һ��ֵ
%��ΪҪ��ž������Զ���һ��1*K�İ�Ԫ
%A=cell��1��K����
%A{:}={[0]};�����а�Ԫ��ʼ��
%http://www.ilovematlab.cn/thread-53246-1-1.html
A=cell(1,K);
WW=cell(1,K);
%L=cell(1,K);

%��Ϊ�Ǹ������£�����ֻ����ÿһ����ͼ�ṹ��û�и������ͼ��L���������Ҫֻ��ı�ѭ����

%��Ϊlabel�����������fea����ʱһһ��Ӧ�ģ�����Ҫ��ȡÿһ�����Ϣ����ͨ���������ʵ��
labelA=[];
labelA=[(1:mFea)',label];
for i=1:K
    for j=1:mFea
        if labelA[j,2]==i;  %�ҳ�label����Ӧ���У�
    A{1,i}(j,:)=fea(j,:);    %��ȡ��һ�е�������Ϣ
        end
    end
    WW{1,i}=constructW(A{1,i},options);
%     if alpha > 0
%         %�������������ľ�����ת�ú�ġ�
%     WW{1,i} = alpha*WW{1,i};
%     DCol = full(sum(WW{1,i},2));
%     D = spdiags(DCol,0,nSmp,nSmp);
%     L{1,i} = D - WW{1,i};
%    else
%     L{1,i} = [];
%    end
end
    
    
  
    
   
    