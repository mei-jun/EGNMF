%���ι滮
function[R,fval]=QP(L,V,K)
%%WW  ��ʾͼ����ڽӾ��������ǰ�Ԫ��
%L   ��ʾ�����L=D-WW,������˹����
%V   ����ֽ��е�X=UV��  ��Ϊ���ι滮�漰��Tr(V'LV)
% K  ��ʾ���пռ���Ǳ�صĿ���ͼ�ĸ���

%R   ������Сֵ��ÿ��ri��ȡֵ    ����ΪK*1
%fval   Ŀ�꺯��ֵ����С��ʱ��)
%  exitflag�����������Ƿ�������output�Ƿ��ذ����Ż���Ϣ�Ľṹ��Lambda�Ƿ��ؽ�x������������ճ��ӵĲ�����
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

    



