function H = SimplexFunction(Htmp, H, m)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here 
% m is the number of agents

%Number of Parameters
n = m*m;

%Number of Artificial parameters
v = m*2;

b = ones(v,1);
    
% ******************************************************************** %
% Iterate through the columns of Htmp
for e =1:size(Htmp,2)

    % Check if the column e is present in the base
    flag = 1;

    for i = 1:v
        if (Htmp(:,e) - H(:,i+m)) == zeros(v+1,1)
            flag = 0;
        end
    end

    if flag

        B = H(1:v,m+1:m+v);

        r_e = [Htmp(v+1,e)-Htmp(1:v,e)'*inv(B)'*H(v+1,m+1:m+v)' Htmp(1:v,e)'*inv(B)'];

        % Lexsort
        k = 1;
        while r_e(k) == 0
            k = k +1;
        end
        % Lexicographically ratio test
        if r_e(k) < 0

            % Numerator and denominator for the lexmin
            D = B\Htmp(1:v,e);
            N = [B\b inv(B)];

            % selection of the j-th row of base B
            L = zeros(v,v+1);

            for j = 1:v

                % check if the denominator is different than zero         
                if D(j) > 0               
                    L(j,:) = N(j,:)/D(j);
                else
                    L(j,:) = zeros(1,v+1);
                end   
            end

            % Find the lexmin
            j = 1;
            [L, index] = sortrows(L,'ascend');

            while L(j,:) == 0
                j = j + 1;
            end

            % store the minimum
            l_min = L(j,:);
            j = index(j);

            if isempty(l_min) == 0
                H(:,j+m) = Htmp(:,e);
                %H = sortrows(H', 'descend')';
            end
        end
    end

end 
  
end