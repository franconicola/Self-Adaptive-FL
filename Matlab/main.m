clc;
clear;
close all;

%% Simplex definition

% Maximum number 
M = 10000;

%Number of Agents
m = 50;

%Number of Parameters
n = m*m;

%Number of Artificial parameters
v = m*2;

%Number of iteration
IT = 55;

% Constraints
c = randi([1 10],[n 1]);
c = [c; M*ones(v,1)]';

% The problem matrix
H = Hgeneration(m,c);

% the Base
B = [ eye(v); M*ones(1,v)];

% Agent's matrices
H_tot = zeros((v+1)*m,m+v);
for i = 1:m
    H_tot(i +(i-1)*v:v+i+(i-1)*v,:) = [H(:,1+(i-1)*m:m+(i-1)*m) B];
end

red_cost = zeros(IT,m);

% ******************************************************************** %
% vector iterator doesn't present in base B
for e =1:IT
    for i =1:m
        if( i == m)
            Htemp = sortrows([H_tot(i +(i-1)*v:v+i+(i-1)*v,:)'; H_tot(1:v+1,m+1:m+v)'],'descend')';
        else
            Htemp = sortrows([H_tot(i +(i-1)*v:v+i+(i-1)*v,:)'; H_tot(i+1+i*v:v+i+1+i*v,m+1:m+v)'],'descend')';
        end
        
        H_tot(i +(i-1)*v:v+i+(i-1)*v,:) = SimplexFunction(Htemp, H_tot(i +(i-1)*v:v+i+(i-1)*v,:), m);
        H_tot(i +(i-1)*v:v+i+(i-1)*v,m+1:m+v) = sortrows(H_tot(i +(i-1)*v:v+i+(i-1)*v,m+1:m+v)','descend')';
        red_cost(e,i) = ones(1,v)*(inv(H_tot(1 +(i-1)*v:v+(i-1)*v,m+1:m+v)')*H_tot(v+1,m+1:m+v)');
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = 1:m
%     
% end

t = 1:IT;
figure
plot(t,red_cost(:,1),'--gs','LineWidth',2,'MarkerSize',5,'MarkerEdgeColor','b','MarkerFaceColor',[0.5,0.5,0.5])
hold on
















