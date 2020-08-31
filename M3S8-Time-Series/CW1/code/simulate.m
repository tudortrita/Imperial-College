function [Y1,Y2,Y3] = simulate(Nb)
%Function for computing vectors Yi
Y = zeros(1,Nb+3);
Y(1)=100;
for i = 2:(Nb+3)
    Y(i)=0.5*Y(i-1)+normrnd(0,1);
end
Y = Y(Nb+1:end);
Y1=Y(1);
Y2=Y(2);
Y3=Y(3);
end

