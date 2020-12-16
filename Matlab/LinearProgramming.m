function [subsetsOrder] = LinearProgramming(numDevices, loss)
% LINEAR PROGRAMMING 
% In this section, we define the entire linear programming set-up


% Maximum number 
M = sum(loss);

% Number of devices
m = numDevices;

% Number of artificial parameters
v = m*2;

% Iteration number of communication
iter_of_comm = m*m;

% Cost vector (L)
cost = [loss M*ones(1, v)];

% The problem matrix
H = Hgeneration(m, cost);

% the Base
B = [ eye(v); M*ones(1,v)];


% Devices's H matrices
H_tot = zeros((v+1)*m,m+v);

for i = 1:m
    H_tot(i +(i-1)*v:v+i+(i-1)*v,:) = [H(:,1+(i-1)*m:m+(i-1)*m) B];
end


% Reduced cost 
red_cost = zeros(iter_of_comm,m);

% vector iterator doesn't present in base B
for e = 1:iter_of_comm
    for i =1:m
        
        % Ring graph: 
        % The last device receives the columns from the first one.
        if( i == m)
            Htemp = sortrows([H_tot(i +(i-1)*v:v+i+(i-1)*v,:)'; 
                H_tot(1:v+1,m+1:m+v)'],'descend')';
        
        % Each device receives the columns from the next one. 
        else
            Htemp = sortrows([H_tot(i +(i-1)*v:v+i+(i-1)*v,:)'; 
                H_tot(i+1+i*v:v+i+1+i*v,m+1:m+v)'],'descend')';
        end
        
        % Simplex 
        H_tot(i +(i-1)*v:v+i+(i-1)*v,:) = SimplexFunction(Htemp, ...
            H_tot(i +(i-1)*v:v+i+(i-1)*v,:), m);
        
        H_tot(i +(i-1)*v:v+i+(i-1)*v,m+1:m+v) = ...
            sortrows(H_tot(i +(i-1)*v:v+i+(i-1)*v,m+1:m+v)','descend')';
        
        red_cost(e,i) = ones(1,v)*...
        (inv(H_tot(1 +(i-1)*v:v+(i-1)*v,m+1:m+v)')*H_tot(v+1,m+1:m+v)');
    end
end

% Extract the basis
basis = H_tot(1:v+1, m+1:m+v);

% Find the solution
solution = basis(1:v, :)\ones(v, 1);

% How to analyze the solution?
subsetsOrder = zeros(1, m);

% Iterate through the solution vector
for i = 1:v
    % Each one corresponds to a column in the basis
    if solution(i) == 1
        % Iterate through the column in the basis
        for j = 1:m
            % Each one correspondes to the new subset
            if basis(m + j, i) == 1
                subsetsOrder(ceil(i/2)) = j;
            end
        end
    end
    
end
