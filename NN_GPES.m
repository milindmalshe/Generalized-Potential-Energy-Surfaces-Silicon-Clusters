
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% This code is for the Development of Generalized Potenial%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of epochs
epochs=1000;

% Goal to reach
goal=0;

% Gradient 'cutoff'
min_grad = 1e-10;


mu = 0.001;
mu_dec = 0.1;
mu_inc = 10;
mu_max = 1e10;


% Defining NNs for fc(r), f(r) and f(theta)

net_fc = newff([1.2 4.7],[2 1],{'purelin' 'purelin'},'trainlm');
net_fc.iw{1,1}= [-1.4; 1.4];
net_fc.lw{2,1}= [-2.0 1.0];
net_fc.b{1,1}= [2.5; 1.5];
net_fc.b{2,1}= [0.6];


net_fr = newff([1.2 4.7],[2 1],{'purelin' 'purelin'},'trainlm');
net_fr.iw{1,1}= [-1.4; 1.4];
net_fr.lw{2,1}= [-2.0 1.0];
net_fr.b{1,1}= [2.5; 1.5];
net_fr.b{2,1}= [0.6];


net_ftheta = newff([1.2 4.7; 1.2 4.7 ;1.2 4.7 ],[2 1],{'purelin' 'purelin'},'trainlm');
net_ftheta.iw{1,1}= [-1.4 1.4 1.2; 1.4 -1.4 1.2];
net_ftheta.lw{2,1}= [-2.0 1.0];
net_ftheta.b{1,1}= [2.5; 1.5];
net_ftheta.b{2,1}= [0.6];


data = xlsread('si_clust_5.xlsx');

%no of data pts
Q=length(data);
% Q=21;

%in future clust_size and type will be read from the data file
total_columns=length(data(1,:));

% column 1 in the data file -> cluster size
max_clust_size=max(data(1:Q,1)); %5


% For now, all the atoms are of same type. In future it will be for different types
type(1:max_clust_size)=2;


for iQ=1:1:Q
        
    clust_size(iQ)=data(iQ,1);     %#ok<AGROW>
    colcount=2;
    for it=1:1:clust_size(iQ)
        for jt=it:1:clust_size(iQ)
            if jt>it
                
                rs(it,jt,iQ)=data(iQ,colcount); %#ok<AGROW>
                rs(jt,it,iQ)=rs(it,jt,iQ); %#ok<AGROW>
                colcount=colcount+1;
            end
            
        end
    end
    V(iQ)=data(iQ,total_columns); %#ok<AGROW>
end


X_fc = getx(net_fc);

X_fr = getx(net_fr);

X_ftheta = getx(net_ftheta);
S1=2;

% X(1:3*S1+1)=X_fc;
% X(3*S1+1+1:3*S1+1+3*S1+1)=X_fr(1:3*S1+1);
% X(3*S1+1+3*S1+1+1:3*S1+1+3*S1+1+5*S1+1)=X_ftheta(1:5*S1+1);

% X(1:3*S1+1)=X_fc;
X(1:3*S1+1,1)=X_fr(1:3*S1+1);
X(3*S1+1+1:3*S1+1+5*S1+1,1)=X_ftheta(1:5*S1+1);


% numParameters=length(X_fc)+length(X_fr)+length(X_ftheta);
numParameters=length(X_fr)+length(X_ftheta);

ii = sparse(1:numParameters,1:numParameters,ones(1,numParameters));%%%

[perf,Ex, Vhat] = calcperf_GPES(net_fc, net_fr, net_ftheta, rs, V, Q, clust_size, type);


[gXt,jjt,normgX]=calcjejj_GPES(net_fc, net_fr, net_ftheta, rs, clust_size,Q,type,Ex);%%%%


for epoch=0:epochs
    
    [je,jj,normgX]=calcjejj_GPES(net_fc, net_fr, net_ftheta, rs, clust_size,Q,type,Ex);%%%%
    
    
    normgX;
  
    if(isnan(normgX) == 1)
        normgX
    end

    % Training Record
    epochPlus1 = epoch+1;
    tr.perf(epochPlus1) = perf;
    tr.mu(epochPlus1) = mu;
    tr.gradient(epochPlus1) = normgX;

    % Stopping Criteria
    %   currentTime = etime(clock,startTime);
    if (perf <= goal)
        stop = 'Performance goal met.';
    elseif (epoch == epochs)
        stop = 'Maximum epoch reached, performance goal was not met.';
        %   elseif (currentTime > time)
        %     stop = 'Maximum time elapsed, performance goal was not met.';
    elseif (normgX < min_grad)
        stop = 'Minimum gradient reached, performance goal was not met.';
    elseif (mu > mu_max)
        stop = 'Maximum MU reached, performance goal was not met.';
        %   elseif (doValidation) & (VV.numFail > max_fail)
        %     stop = 'Validation stop.';
        %   elseif flag_stop
        %     stop = 'User stop.';
    end

    % Progress
    %   if isfinite(show) & (~rem(epoch,show) | length(stop))
    %     fprintf('%s%s%s',this,'-',gradientFcn);
    if isfinite(epochs) fprintf(', Epoch %g/%g',epoch, epochs); end
    %   if isfinite(time) fprintf(', Time %4.1f%%',currentTime/time*100); end
    
    if isfinite(goal) fprintf(', %s %g/%g',upper(net_fc.performFcn),perf,goal); end
    
    if isfinite(goal) fprintf(', %s %g/%g',upper(net_fr.performFcn),perf,goal); end
    
    if isfinite(goal) fprintf(', %s %g/%g',upper(net_ftheta.performFcn),perf,goal); end
    
    
    if isfinite(min_grad) fprintf(', Gradient %g/%g',normgX,min_grad); end
    fprintf('\n')
    %   flag_stop=plotperf(tr,goal,this,epoch);
    %     if length(stop) fprintf('%s, %s\n\n',this,stop); end
    %   end

    % Stop when criteria indicate its time
    %   if length(stop)
    %     if (doValidation)
    %     net = VV.net;
    %   end
    %     break
    %   end

    % Levenberg Marquardt
    while (mu <= mu_max)
        % CHECK FOR SINGULAR MATRIX
        [msgstr,msgid] = lastwarn;
        lastwarn('MATLAB:nothing','MATLAB:nothing')
        warnstate = warning('off','all');
        dX = -(jj+ii*mu) \ je;
        [msgstr1,msgid1] = lastwarn;
        flag_inv = isequal(msgid1,'MATLAB:nothing');
        if flag_inv, lastwarn(msgstr,msgid); end;
        warning(warnstate)
        X2 = X + dX;

%         %     X2(1:2,1) = 0;%%%%%*********************Setting NN weights to 0
%         %     dX(1:2,1) = 0;
% 
%         % %%% Avoiding beta <0
%         if X2(7,1)<1e-15
%             X2(7,1);
%             X2(7,1)=1e-15;
%         end

%         net2_fc = setx(net_fc,X2(1:3*S1+1));
%         net2_fr = setx(net_fr,X2(3*S1+1+1:3*S1+1+3*S1+1));
%         net2_ftheta = setx(net_ftheta,X2(3*S1+1+3*S1+1+1:3*S1+1+3*S1+1+5*S1+1));

        net2_fc = net_fc;
        net2_fr = setx(net_fr,X2(1:3*S1+1+3*S1+1,1));
        net2_ftheta = setx(net_ftheta,X2(3*S1+1+1:3*S1+1+5*S1+1,1));
        
%         for ijk=1:1:9
%             if ijk<3
%                 param(2,2,2,ijk+2)=X2(ijk+4);
%             else
%                 param(2,2,2,ijk+4)=X2(ijk+4);
%             end
%         end
        [perf2,Ex, Vhat] = calcperf_GPES(net2_fc, net2_fr, net2_ftheta, rs, V, Q, clust_size, type);
        V-Vhat;
        if (perf2 < perf) && flag_inv
%             X = X2; net = net2; %Zb = Zb2; Zi = Zi2; Zl = Zl2;
            X = X2; net_fc = net2_fc; net_fr=net2_fr; net_ftheta=net2_ftheta;%Zb = Zb2; Zi = Zi2; Zl = Zl2;
            %N = N2; Ac = Ac2; El = El2;
            perf = perf2;
            mu = mu * mu_dec;
            if (mu < 1e-20)
                mu = 1e-20;
            end
            break   % Must be after the IF
        end
        mu = mu * mu_inc;
    end

    % Validation
    %   if (doValidation)
    %       [vperf,Ex, Vhat] = calcperf_NN_Tersoff(net2,rs,V,Q,param,clust_size,type);%vperf = calcperf(net,X,VV.Pd,VV.Tl,VV.Ai,VV.Q,VV.TS);
    %       if (vperf < VV.perf)
    %           VV.perf = vperf; VV.net = net; VV.numFail = 0;
    %       elseif (vperf > VV.perf)
    %           VV.numFail = VV.numFail + 1;
    %       end
    %   end



end


