clear all

%% Simulate data
% Fix Random number generator seed
seed=98986;
rng(seed);

n=700; %sample size
d=3; % number of dependent variables
p=1; % number of continuous covariates
q=3; % number of discrete covariates
Gq= 2*ones(1,q); %number of categories for each discrete variable

%Initialize covariates
x=zeros(n,p+q); 
x_noDisc=zeros(n,p);

%x_1: Continuous covariate: (Mimics Age at interview)
x_noDisc(:,1) = rand(n,1)*15+15;
x(:,1)=floor(x_noDisc);

%x_p+1 and x_p+2: dummies (Mimic Region)
h=1;
probs1=[0.5,0.3,0.2]';
aux=mnrnd(1,probs1,n);
x(:,p+h:p+h+1)=aux(:,1:2)+1;

% x_2: dummy (Mimics urban/Rural)
probs2=[0.4,0.6];
aux=mnrnd(1,probs2,n);
x(:,p+q)=aux(:,1)+1;

%Initialize dependent variables
z=zeros(n,d); 
z_noDisc=zeros(n,d);

% Error distribution: mixture of normals to provide heavy tail
ed=@(x).9*normpdf(x,-15/90,0.5)+0.1*normpdf(x,1.5,0.75);

% First variable: discretized continuous (age at sex)
IUR=x(:,p+q)==2; % urban indicator
IReg=x(:,p+1)==2; % For region 1
e1=zeros(n,1); % errors to be used to simulate second variable
% If Region 1 and Urban
I=IReg&IUR;
g1 = @(x)0.0001*x.^3-0.0695*x.^2+3.83*x-30.584;
z_noDisc(I,1) = g1(x_noDisc(I,1));
aux=rand(sum(I),1)<.9;
e1(I)=aux.*normrnd(-15/90,0.5,sum(I),1)+(1-aux).*normrnd(1.5,0.75,sum(I),1);
z_noDisc(I,1) =z_noDisc(I,1)+e1(I);
% If Region 2 and 3 and urban
g2 = @(x)-0.057*x.^2+3.08*x-21.247;
I=~IReg&IUR;
z_noDisc(I,1) = g2(x_noDisc(I,1));
aux=rand(sum(I),1)<.9;
e1(I)=aux.*normrnd(-15/90,0.5,sum(I),1)+(1-aux).*normrnd(1.5,0.75,sum(I),1);
z_noDisc(I,1) =z_noDisc(I,1)+e1(I);
% If Region 1 and rural
g3 =  @(x)((23-15)/(30-15))*x+7;
I=IReg&~IUR;
z_noDisc(I,1) = g3(x_noDisc(I,1));
aux=rand(sum(I),1)<.9;
e1(I)=aux.*normrnd(-15/90,0.5,sum(I),1)+(1-aux).*normrnd(1.5,0.75,sum(I),1);
z_noDisc(I,1) =z_noDisc(I,1)+e1(I);
% If Region 2 and 3 and rural
g4 = @(x)((20-15)/(30-15))*x+10; 
I=~IReg&~IUR;
z_noDisc(I,1) = g4(x_noDisc(I,1));
aux=rand(sum(I),1)<.9;
e1(I)=aux.*normrnd(-15/90,0.5,sum(I),1)+(1-aux).*normrnd(1.5,0.75,sum(I),1);
z_noDisc(I,1) =z_noDisc(I,1)+e1(I);

% Second variable: discretized continuous (age at union)
nIUR=sum(IUR);
% If rural
g5 = @(x)0.5*x+8;
g6 = @(x)7.5./x;
aux=rand(n-nIUR,1)<.9;
z_noDisc(~IUR,2) = g5(x_noDisc(~IUR,1))+0.75*e1(~IUR)+aux.*normrnd(-15/90,g6(x_noDisc(~IUR,1)),n-nIUR,1)+(1-aux).*normrnd(1.5,g6(x_noDisc(~IUR,1)),n-nIUR,1);
% If urban
g7 = @(x)-0.056*x.^2+3.08*x-18;
aux=rand(nIUR,1)<.9;
z_noDisc(IUR,2) = g7(x_noDisc(IUR,1))+0.75*e1(IUR)+aux.*normrnd(-15/90,0.4,nIUR,1)+(1-aux).*normrnd(1.5,0.75,nIUR,1);

% Censoring: censored if age at sex debut/union is bigger than age at
% interview
c=ones(n,2);
c(z_noDisc(:,1)>x_noDisc(:,1),1)=0;
c(z_noDisc(:,2)>x_noDisc(:,1),2)=0;
z(:,1)=floor(z_noDisc(:,1));
z(z_noDisc(:,1)>x_noDisc(:,1),1)=NaN;
z(:,2)=floor(z_noDisc(:,2));
z(z_noDisc(:,2)>x_noDisc(:,1),2)=NaN;

% Third variable: respondent work (0/1)
g9 = @(x)normcdf((x-18)/6);
pp = g9(x_noDisc(:,1));
z(:,3) = (rand(n,1)<pp);

save('simulateddata.mat','z','c','x','n','p','q');
