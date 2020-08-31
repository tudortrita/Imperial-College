function [Xtdiff] = diff_operator(Xt)
%Function for ocmputing difference operator on time series

%Backwards shift
Xtmin1 = Xt(2:end);

%Calculating first difference
Xtdiff = Xt(1:(end-1)) - Xtmin1;
end

