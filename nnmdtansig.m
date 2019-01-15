function d = nnmdtansig(a,d,w)
%  NNMDLOG Logistic Delta Function for Marquardt.
%          Returns the delta values for a layer of
%          log-sigmoid neurons for use with Marquardt
%          backpropagation.
%==================================================================

[s1,q]=size(a);

if nargin == 1
  d = -kron(((ones-a).*a),ones(1,s1)).*kron(ones(1,s1),eye(s1));
else
    %   d = ((ones-a).*a).*(w'*d);
    d = ((ones-a.^2)).*(w'*d);
end
