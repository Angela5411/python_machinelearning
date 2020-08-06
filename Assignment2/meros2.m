s1=[1 1 1];
s2=[-1 1 1];
s3=[1 -1 1];
s4=[-1 -1 1];
A=[sum(s1.*s1) sum(s2.*s1) sum(s3.*s1) sum(s4.*s1);
   sum(s1.*s2) sum(s2.*s2) sum(s3.*s2) sum(s4.*s2);
   sum(s1.*s3) sum(s2.*s3) sum(s3.*s3) sum(s4.*s3);
   sum(s1.*s4) sum(s2.*s4) sum(s3.*s4) sum(s4.*s4)];
Y=[1 1 -1 -1];
X=Y/A
p=X(1);
q=X(2);
r=X(3);
s=X(4);
W=[p*s1+q*s2+r*s3+s*s4]
x=-2:0.02:2;
y=((-W(1)*x)-W(3))/W(2);
figure()
plot(x,y,'b-')
xlim([-2 2])
ylim([-2 2])
 hold on
 plot(1,1,'r+')
 hold on
 plot(-1,1,'r+')
 hold on
 plot(1,-1,'b.')
 hold on
 plot(-1,-1,'b.')