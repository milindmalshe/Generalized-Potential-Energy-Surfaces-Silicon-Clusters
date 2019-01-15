function d = nnmdlin(a,d,w)
%NNMDLIN  Logistic Delta Function for Marquardt.
%         Returns the delta values for a layer of
%         linear neurons.
%==================================================================

[na,ma]=size(a);

if nargin == 1
  d=-kron(ones(1,ma),eye(na));
else
  d = w'*d;
end

