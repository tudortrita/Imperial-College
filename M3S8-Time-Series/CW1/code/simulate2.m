function [X] = simulate2(Nb,N)
%Function for computing vectors Yi

X = zeros(1,Nb+N);
Y = zeros(1,Nb+3);

for t = 2:(Nb+N)
    Y(t)=0.5*Y(t-1)+normrnd(0,1);
    X(t)= 1 + 0.02*t + Y(t);
end
X = X(Nb+1:end);
end


