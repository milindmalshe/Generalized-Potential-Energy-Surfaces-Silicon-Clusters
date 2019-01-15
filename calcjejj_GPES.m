%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% This code is for the Development of Generalized Potenial%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [JE,JtJ,normJE]=calcjejj_GPES(net_fc, net_fr, net_ftheta, rs, clust_size, Q, type, Ex)

% There will be 3 Jacobian matrices, one each for fc, fr, and ftheta
% numNetParam_fc=length(getx(net_fc));
% 
% numNetParam_fr=length(getx(net_fr));
% 
% numNetParam_theta=length(getx(net_ftheta));



%%%% Jacobian matrix for fr
%J_fr will have dimensions of Q x ( 3N + 1 )
% First layer W (N x 1), b (N x 1), Second Layer W (1 x N ), b (1x1)
W1 = net_fr.IW{1,1};
W2 = net_fr.LW{2,1};
B1 = net_fr.b{1};
B2 = net_fr.b{2};



S1 = 2;
S2 = 1;

% %%%%%%%
for iQ = 1:Q

    ciQ = 0;

    for i=1:1:clust_size(iQ)
        for j=1:1:clust_size(iQ)
            if j>i%j~= i %
                ciQ = ciQ+1;
                r_fr(ciQ) = rs(i,j,iQ);

            end
        end
    end




    A1 = purelin(W1*r_fr + B1*ones(1,(clust_size(iQ)*(clust_size(iQ)-1)/2)*1));
    A2 = W2*A1 + B2*ones(1,(clust_size(iQ)*(clust_size(iQ)-1)/2)*1);


    % FIND JACOBIAN
    A1 = kron(A1,ones(1,S2));
    D2 = nnmdlin(A2);


    D1 = nnmdlin(A1,D2,W2);
    jac1 = nnlmarq(kron(r_fr,ones(1,S2)),D1);
    jac2 = nnlmarq(A1,D2);

    jac_fr = [jac1,D1',jac2,D2']; % Make jacbian for current iQ configuration


    for j= 1: 3*S1+1
        jac(1,j)= 0;
        for i = 1:(clust_size(iQ)*(clust_size(iQ)-1)/2)
            jac(1,j) = jac(1,j)+jac_fr(i,j);  % sum jacobian for current iQ configuration along all rows to get 1 row

        end
    end

    for j= 1: 3*S1+1
        J_fr(iQ,j)= jac(1,j);
    end

end


% %%%%%%%

%%%%% following code makes intermediate jacobian matrix which could be very big in size
% ciQ = 0;
% for iQ = 1:Q
%     for i=1:1:clust_size
%         for j=1:1:clust_size
%             if j>i
%                 ciQ = ciQ+1;
%                 r_fr(ciQ) = rs(i,j,iQ);
%                 
%             end
%         end
%     end
% 
% end
% 
% 
% A1 = purelin(W1*r_fr + B1*ones(1,(clust_size*(clust_size-1)/2)*Q));
% A2 = W2*A1+B2*ones(1,(clust_size*(clust_size-1)/2)*Q);
% 
% % FIND JACOBIAN
% A1 = kron(A1,ones(1,S2));
% D2 = nnmdlin(A2);
% 
% 
% D1 = nnmdlin(A1,D2,W2);
% jac1 = nnlmarq(kron(r_fr,ones(1,S2)),D1);
% jac2 = nnlmarq(A1,D2);
% 
% jac_fr = [jac1,D1',jac2,D2'];
% 
% 
% % J_fr= 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    


% %%%%% following code makes intermediate jacobian matrix which could be very big in size
% % J_theta will have dimensions of Q x ( 5N + 1)
% % First layer W (N x 3), b (N x 1), Second Layer W (1 x N ), b (1x1)
% 
% W1 = net_ftheta.IW{1,1};
% W2 = net_ftheta.LW{2,1};
% B1 = net_ftheta.b{1};
% B2 = net_ftheta.b{2};
% 
% 
% 
% S1 = 2;
% S2 = 1;
% 
% 
% 
% ciQ = 0;
% for iQ = 1:Q
%     for i=1:1:clust_size
%         for j=1:1:clust_size
%             for k=1:1:clust_size
%                 if (j>i & k>i & k>j)
%                     ciQ = ciQ+1;
%                     r_ftheta(1,ciQ) = rs(i,j,iQ);
%                     r_ftheta(2,ciQ) = rs(i,k,iQ);
%                     r_ftheta(3,ciQ) = rs(j,k,iQ);
% 
%                 end
%             end
%         end
%     end
% 
% end
% 
% 
% 
% 
% A1 = purelin(W1*r_ftheta + B1*ones(1,(clust_size*(clust_size-1)*(clust_size-2)/6)*Q));
% A2 = W2*A1+B2*ones(1,(clust_size*(clust_size-1)*(clust_size-2)/6)*Q);
% 
% % FIND JACOBIAN
% A1 = kron(A1,ones(1,S2));
% D2 = nnmdlin(A2);
% 
% 
% D1 = nnmdlin(A1,D2,W2);
% jac1 = nnlmarq(kron(r_ftheta,ones(1,S2)),D1);
% jac2 = nnlmarq(A1,D2);
% 
% jac_ftheta = [jac1,D1',jac2,D2'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% J_theta will have dimensions of Q x ( 5N + 1)
% First layer W (N x 3), b (N x 1), Second Layer W (1 x N ), b (1x1)

W1 = net_ftheta.IW{1,1};
W2 = net_ftheta.LW{2,1};
B1 = net_ftheta.b{1};
B2 = net_ftheta.b{2};



S1 = 2;
S2 = 1;




for iQ = 1:Q
    ciQ = 0;
    for i=1:1:clust_size(iQ)
        for j=1:1:clust_size(iQ)
            for k=1:1:clust_size(iQ)
                if (j>i & k>i & k>j)
                    ciQ = ciQ+1;
                    r_ftheta(1,ciQ) = rs(i,j,iQ);
                    r_ftheta(2,ciQ) = rs(i,k,iQ);
                    r_ftheta(3,ciQ) = rs(j,k,iQ);

                end
            end
        end
    end

    A1 = purelin(W1*r_ftheta + B1*ones(1,(clust_size(iQ)*(clust_size(iQ)-1)*(clust_size(iQ)-2)/6)*1));
    A2 = W2*A1+B2*ones(1,(clust_size(iQ)*(clust_size(iQ)-1)*(clust_size(iQ)-2)/6)*1);

    % FIND JACOBIAN
    A1 = kron(A1,ones(1,S2));
    D2 = nnmdlin(A2);


    D1 = nnmdlin(A1,D2,W2);
    jac1 = nnlmarq(kron(r_ftheta,ones(1,S2)),D1);
    jac2 = nnlmarq(A1,D2);

    jac_ftheta = [jac1,D1',jac2,D2'];


    %%%%
    for j= 1: 5*S1+1
        jac(1,j)= 0;
        for i = 1:(clust_size(iQ)*(clust_size(iQ)-1)*(clust_size(iQ)-2)/6)
            jac(1,j) = jac(1,j)+jac_ftheta(i,j);  % sum jacobian for current iQ configuration along all rows to get 1 row

        end
    end

    for j= 1: 5*S1+1
        J_ftheta(iQ,j)= jac(1,j);
    end



end





%%%Make cobined Jacobian matrix
J = zeros(Q,(3*S1+1)+(5*S1+1) );

J(:,1:3*S1+1)=J_fr;
J(:,(3*S1+1)+1:(3*S1+1)+(5*S1+1))=J_ftheta;

JE=J'*Ex;
JtJ=J'*J;
normJE=sqrt(JE'*JE);


% %%%Make cobined Jacobian matrix
% J = zeros(Q,(3*S1+1)+(3*S1+1)+(5*S1+1) );
% 
% J(:,3*S1+1+1:3*S1+1+3*S1+1)=J_fr;
% J(:,3*S1+1+3*S1+1+1:3*S1+1+3*S1+1+5*S1+1)=J_ftheta;
% 
% JE=J'*Ex;
% JtJ=J'*J;
% normJE=sqrt(JE'*JE);




i=1; %dummy line for breakpoint




