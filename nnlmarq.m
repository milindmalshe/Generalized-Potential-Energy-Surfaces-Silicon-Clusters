function jac = nnlmarq(p,d)
%NNLMARQ  Marquardt Backpropagation Learning Rule

if nargin ~= 2
  error('Wrong number of arguments.');
end

[s,q]=size(d);
[r,q]=size(p);

jac=kron(p',ones(1,s)).*kron(ones(1,r),d');

