function Hgen = Hgeneration(m,c)
% H generation
% m agents
% c vector of constraints
H = zeros(2*m+1,m*m);

for i = 1:2*m+1 
    k = 1;
    
    for j = 1:m*m

        if (i <= m)

            if( j >  (i-1)*m && j <=  i*m)
                H(i,j) = 1;
            else
                H(i,j) = 0;
            end

        elseif ( i > m && i <= 2*m)

            if (i - m == j)
                H(i,j) = 1;
            elseif (j == m*k + i - m)
                H(i,j) = 1;
                k = k +1;
            else
                H(i,j) = 0;	
            end

        elseif(i > 2*m)
            H(i,j) = c(j);
        end
    end
end

Hgen = H;
end

