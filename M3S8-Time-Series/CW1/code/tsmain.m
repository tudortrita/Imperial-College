% Time Series Coursework 1 Main Code:
% Tudor Trita Trita
% CID: 01199397

M=1000000;
Ymat = zeros(M,3);
Covempirical = zeros(1,M);
Covtheory = [4/3,2/3,1/3;2/3,4/3,2/3;1/3,2/3,4/3];
Norm = zeros(1);

for i = 1:20    %REPEAT 20 TIMES
    %Computing vector Y 
    for j = 1:M
        [Ymat(j,1),Ymat(j,2),Ymat(j,3)]=simulate(i);
    end 
    
    %Computing empirical covariance
    covariance = cov(Ymat);
    s0 = covariance(1,1);
    s1 = covariance(1,2);
    s2 = covariance(1,3);
    Covempirical = [s0,s1,s2;s1,s0,s1;s2,s1,s0]; 
    
    %Compute NORM for each Y
    mat = Covtheory-Covempirical;
    Norm(i)= norm(mat,'fro');
end

plot(1:20,Norm)


