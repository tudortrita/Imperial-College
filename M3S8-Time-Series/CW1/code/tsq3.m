%Time Series Coursework 1 Question 3 Code
%Tudor Trita Trita
%CID: 01199397

%Burn-in Time
Nb = 50;
M = 10000;
s0emat = zeros(M,1);
s1emat = zeros(M,1);
s2emat = zeros(M,1);
s0varmat = zeros(19,1);
s1varmat = zeros(19,1);
s2varmat = zeros(19,1);
s0biasmat = zeros(19,1);
s1biasmat = zeros(19,1);
s2biasmat = zeros(19,1);
s0th = 24;
s1th = 16;
s2th = 4;


for N = 100:50:1000
    s0e = 0;
    s1e = 0;
    s2e = 0;
    for i = 1:M
        Xt = simulate2(Nb,N);
        Xtdetrend1 = diff_operator(Xt); %Length N-1
        Xtdetrendfull = diff_operator(Xtdetrend1); %Length N-2
    
        %Estimating s's
      
        for t = 1:N-2
            iter = Xtdetrendfull(t)*Xtdetrendfull(t);
            s0e = s0e + iter;
        end
        s0e = s0e/N;
    
        for t = 1:N-3
            iter = Xtdetrendfull(t)*Xtdetrendfull(t+1);
            s1e = s1e + iter;
        end
        s1e = s1e/N;
    
        for t = 1:N-4
            iter = Xtdetrendfull(t)*Xtdetrendfull(t+2);
            s2e = s2e + iter;
        end
        s2e = s2e/N;
        
        s0emat(i)=s0e;
        s1emat(i)=s1e;
        s2emat(i)=s2e;
    end
    %Compute Variance of estimators:
    s0var = var(s0emat);
    s1var = var(s1emat);
    s2var = var(s2emat);  
    r = (N-50)/50;
    s0varmat(r)=s0var;
    s1varmat(r)=s1var;
    s2varmat(r)=s2var;
end