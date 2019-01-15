%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% This code is for the Development of Generalized Potenial%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [perf,Ex,Vhat]=calcperf_GPES(net_fc, net_fr, net_ftheta, rs, V, Q, clust_size, type)

perf=0.0;

for iQ=1:1:Q
    
    %converting rs to my format
    for i=1:1:clust_size(iQ)
         for j=1:1:clust_size(iQ)
             if i~=j
                 r(i,j)=rs(i,j,iQ); %#ok<AGROW>
             end
         end
    end
    
    
    %calculating potential energy for iQ configuration
    Vhat_iQ=0.0;
    totalTriangles=0;
    fr=0.0;
    ftheta=0.0;
    for i=1:1:clust_size(iQ)
        for j=1:1:clust_size(iQ)
            if j>i
                rij=r(i,j);
                fc=1;%sim(net_fc,rij);
                fr=fr+sim(net_fr,r(i,j));
            end
            for k=1:1:clust_size(iQ)
                if (j>i && k>i && k>j)
                    rij=r(i,j);
                    rik=r(i,k);
                    rjk=r(j,k);
                    ftheta=ftheta+sim(net_ftheta, [rij; rik ;rjk]);
                    totalTriangles=totalTriangles+1;
                    
                end
            end
           
        end
    end
    
%     
%     for i=1:1:clust_size(iQ)
%         for j=1:1:clust_size(iQ)
%                                        
%            
%         end
%     end
    
    Vhat_iQ=fc*(fr+ftheta);
    Vhat(iQ)=Vhat_iQ; %#ok<AGROW>
    Ex(iQ,1)=V(iQ)-Vhat(iQ);  %#ok<AGROW>
    %SSE
    perf=perf+Ex(iQ,1)*Ex(iQ,1);
     
                 
    
end

