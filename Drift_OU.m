%%MLPMCMC with diffusion bridge, estimating drift coefficient. 
close all;
clear;
clc;
format long

T = 65;
L = 18;
delta = 2^(-L); 
nh = 1/delta;
nl = T*2^L; 

% parameters 
A = [0.8,0.2;-0.3,0.8];
x0 = randn(2,1)+3;
R = chol(eye(2) * delta);
COV = [1,0.5; 0.5,1];
Sigma = sqrtm(COV);
rho = 0.5;

load('X_18.mat');
NP = 50;
NP_1 = 50;

Lmin = 4;
LP = 6;
% discretization size
hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%Nl =  floor(20 * 2^(2*LP) * hl.^(9/10) + 1000);
Nl = [5000,5000,5000];
%store the acceptance rate
Aln = zeros(LP - Lmin + 1, 1);
Theta_trace = cell(LP - Lmin + 1, 1);
Theta_trace_1 = cell(LP - Lmin,1);
Theta_trace_2 = cell(LP - Lmin,1);

Theta_traceN = cell(LP - Lmin + 1, 1);
Theta_trace_1N = cell(LP - Lmin,1);
Theta_trace_2N = cell(LP - Lmin,1);

%mean of theta over iterations
ML_Theta_trace = cell(LP - Lmin + 1, 1);
%weights for finer and corse level        
H1_trace = cell(LP - Lmin, 1);
H2_trace = cell(LP - Lmin, 1);

for k = 1 : LP - Lmin + 1
    Theta_trace{k, 1} = zeros(Nl(k),4);
    Theta_traceN{k,1} = zeros(Nl(k),4);
    ML_Theta_trace{k, 1} = zeros(Nl(k),4);
end

for i = 1:LP - Lmin
    Theta_trace_1{i,1} = zeros(Nl(i+1),4);
    Theta_trace_2{i,1} = zeros(Nl(i+1),4);

    Theta_trace_1N{i,1} = zeros(Nl(i+1),4);
    Theta_trace_2N{i,1} = zeros(Nl(i+1),4);

    H1_trace{i,1} = zeros(Nl(i+1),1);
    H2_trace{i,1} = zeros(Nl(i+1),1);
end


%initial theta values and parameters
Theta_A = [0.9,0.2,-0.3,0.6];
tic;

Theta_A_p = Theta_A;
Z = PF_DB(Lmin, NP, T, X, Theta_A_p(1),Theta_A_p(2),Theta_A_p(3),Theta_A_p(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);

N_count = 0;
N_count_last = 0;

Sigma_A = 0.4*diag([0.8,0.6,0.6,0.8]);
Sigma_A1 = 0.35*diag([0.8,0.6,0.6,0.8]);

for iter = 1:Nl(1)
 
    if mod(iter, 50) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count/(iter))]);
        disp(['current new AC = ', num2str((N_count - N_count_last)/(mod(iter,50)+1))]);

    end
    if mod(iter, 50) == 0
        N_count_last = N_count;
    end

   
    Theta_A_prime1 = Theta_A_p;
    Theta_A_prime1 = mvnrnd(Theta_A_prime1, Sigma_A*Sigma_A');
    Theta_A_prime = Theta_A_prime1;

    Z_prime = PF_DB(Lmin, NP, T, X, ...
            Theta_A_prime(1),Theta_A_prime(2),Theta_A_prime(3),Theta_A_prime(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
    lZ_prime = Z_prime;
    l_pos_Theta_A_prime = l_posterior(Theta_A_prime, lZ_prime);
    
    
    alpha = min(0, l_pos_Theta_A_prime - l_pos_Theta_A_p);
    U = log(rand);
    if U < alpha
        Theta_A_p = Theta_A_prime;
        lZ = lZ_prime;
        l_pos_Theta_A_p = l_pos_Theta_A_prime;
        Theta_trace{1, 1}(iter,:) = Theta_A_prime; 
        N_count = N_count + 1;
        
    else
        Theta_trace{1, 1}(iter,:) = Theta_A_p; 
        lZ = PF_DB(Lmin, NP, T, X, Theta_A_p(1),Theta_A_p(2),Theta_A_p(3),Theta_A_p(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
        l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
    end 
end

Aln(1,1) = N_count / Nl(1);
toc;

H1_sum = 0;
H2_sum = 0;
tic;

for l = 1:LP - Lmin 

    level = l + Lmin;
    fprintf('level = %f\n', level);

    Theta_l = mean(Theta_trace{1,1});
    [H1_l, H2_l, G_l] = c_pf_db(level, NP_1, T, X, Theta_l(1),Theta_l(2),Theta_l(3),Theta_l(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
    lG_l = G_l;
    l_pos_theta_l = l_posterior(Theta_l, lG_l);
    
    N_count_l = 0;
    for iter = 1:Nl(l+1)
        
        if mod(iter, 100) == 0
            fprintf('iter = %f\n', iter);
            fprintf('AR = %f\n', N_count_l/iter);
            fprintf('H1 average = %f\n', H1_sum/100);
            fprintf('H2 average = %f\n', H2_sum/100);
            %disp(['average theta = ', num2str(mean(Theta_trace((iter-99):iter-1,:),1))]);
            H1_sum = 0;
            H2_sum = 0;
            toc;
            tic;
        end
            
        Theta_l_prime1 = mvnrnd(Theta_l, Sigma_A1*Sigma_A1');
        Theta_l_prime = Theta_l_prime1;
        [H1_lp, H2_lp, lG_lp] = c_pf_db(level, NP_1, T, X, Theta_l_prime(1),Theta_l_prime(2),Theta_l_prime(3),Theta_l_prime(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
        l_pos_theta_l_prime = l_posterior(Theta_l_prime, lG_lp);
        alpha_l = min(0, l_pos_theta_l_prime - l_pos_theta_l);

        H1_sum = H1_sum + H1_lp;
        H2_sum = H2_sum + H2_lp;

        Ul = log(rand);

        if Ul < alpha_l 
            Theta_l = Theta_l_prime;
            Theta_trace{l+1, 1}(iter,:) = Theta_l_prime;
            lG_l = lG_lp;
            l_pos_theta_l = l_pos_theta_l_prime;
            H1_l = H1_lp;
            H2_l = H2_lp;
            H1_trace{l, 1}(iter,1) = H1_lp;
            H2_trace{l, 1}(iter,1) = H2_lp;
            N_count_l = N_count_l + 1;
        else
            Theta_trace{l+1, 1}(iter,:) = Theta_l; 
            [H1_l, H2_l, lG_l] = c_pf_db(level, NP_1, T, X, Theta_l(1),Theta_l(2),Theta_l(3),Theta_l(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
            l_pos_theta_l = l_posterior(Theta_l, lG_l);
            H1_trace{l, 1}(iter,1) = H1_l;
            H2_trace{l, 1}(iter,1) = H2_l;
        end   
    end

        Aln(l+1,1) = N_count_l / Nl(l+1);
end

toc;

for ll = 1:LP - Lmin
    for i = 1:4      
        Theta_trace_1N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))));
        Theta_trace_2N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H2_trace{ll,1}(:,1)) / sum(exp(H2_trace{ll,1}(:,1))));
        Theta_traceN{ll+1,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))) - exp(H2_trace{ll,1}(:,1))/ sum(exp(H2_trace{ll,1}(:,1))));
        ML_Theta_trace{ll+1,1}(:,i) = cumsum(Theta_traceN{ll+1, 1}(:,i)) ./ (1:Nl(ll+1))';
    end
end

Theta_traceN{1,1} = mean(Theta_trace{1,1});

final_theta = Theta_traceN{1,1};
level_means = zeros(LP-Lmin, 4);
burnin = 1;
for i=1:4
    for j = 1:LP - Lmin
        final_theta(i) = final_theta(i) + sum(Theta_traceN{j+1,1}(burnin:end,i));
        level_means(j,i) = sum(Theta_traceN{j+1,1}(burnin:end,i));
    end
end
Theta_iters = Theta_trace{1,1} +  Theta_traceN{2,1} + Theta_traceN{3,1};
burnin = 1;
niter = 5000;

desired_height = 0.12;
figure_distance = 400;
f = figure;
f.Position = f.Position+[0 -figure_distance 0 figure_distance];

ax = subplot(4,1,1);
plot(burnin:3:niter,Theta_iters(burnin:3:end,1), 'r-',LineWidth=1);
title('A11');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(4,1,2);
plot(burnin:3:niter,Theta_iters(burnin:3:end,2), 'r-',LineWidth=1);
title('A12');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');


ax = subplot(4,1,3);
plot(burnin:3:niter,Theta_iters(burnin:3:end,3), 'r-',LineWidth=1);
title('A21')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(4,1,4);
plot(burnin:3:niter,Theta_iters(burnin:3:end,4), 'r-',LineWidth=1);
title('A22')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');



%particle filter using diffusion bridge
function z = PF_DB(L, NP, T, X,A1,A2,A3,A4,SIG1,SIG2,rho)
    hl = 2^(-L);
    nl = 2^L;
    A  = [A1,A2;A3,A4];
    
    X_est = zeros(NP,2,T+1);
    X_est(:,1,1) = X(1,1) * ones(NP,1);
    X_est(:,2,1) = X(2,1) * ones(NP,1);
    Cova = [SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2];
    Cova_Inv = Cova^(-1);
    lGL1 = zeros(NP,T);
    lGL2 = zeros(NP,T);
    lGL3 = zeros(NP,T);
    lGL = zeros(NP,T);
    lGL_star = zeros(1,T);

    for k = 1:T
        
        XK = zeros(NP,2);
        XKM = zeros(NP,2);
        XKM(:,1) = X_est(:,1,k);
        XKM(:,2) = X_est(:,2,k);
        %update x_{k+1}
        %the first observation is missed
        %simulate x^1_{k+1}conditional on X^2_{k+1} and X_k
        if X(1,k+1) == 0 && X(2,k+1) ~= 0
            X_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            X_est(:,1,k+1) = sqrt((1-rho^2))*SIG1*0.1*randn(NP,1) + X_est(:,1,k) + rho*SIG1/SIG2* (X_est(:,2,k+1)-X_est(:,2,k));
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGL1(:,k) = lG(XK,XKM,Cova);
            %h_hat, conditional density 
            lGL2(:,k) = -lG(X_est(:,1,k+1),X_est(:,1,k) + rho*SIG1/SIG2*(X_est(:,2,k+1)-X_est(:,2,k)),(1-rho^2)*SIG1^2*0.01);
            
            % the integral of L
            for m = 0:nl-1
                lGL3(:,k) = lGL3(:,k) + diag(hl*(-A*X_Bri(:,:,m+1)')' * Cova_Inv*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
            end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);

        %the second observation is missed
        %simulate x^2_{k+1}conditional on X^1_{k+1} and X_k
        elseif X(2,k+1) == 0 && X(1, k+1) ~= 0
            X_est(:,1,k+1) = X(1,k+1) * ones(NP,1);
            %generate the second comnent from f_hat (browanian motion delta_t = 1)
            X_est(:,2,k+1) = sqrt((1-rho^2))*SIG2*0.1*randn(NP,1) + X_est(:,2,k) + rho*SIG2/SIG1* (X_est(:,1,k+1)-X_est(:,1,k));
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGL1(:,k) = lG(XK,XKM,Cova);
            %h_hat, conditional density 
            lGL2(:,k) = -lG(X_est(:,2,k+1),X_est(:,2,k) + rho*SIG2/SIG1*(X_est(:,1,k+1)-X_est(:,1,k)),(1-rho^2)*SIG2^2*0.01);
        
            for m = 0:nl-1
                lGL3(:,k) = lGL3(:,k) + hl*diag((-A*X_Bri(:,:,m+1)')' * Cova_Inv*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
            end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);
        %no missing components
        elseif X(2,k+1) ~= 0 && X(1, k+1) ~= 0
            X_est(:,1,k+1) = X(1,k+1)*ones(NP,1);
            X_est(:,2,k+1) = X(2,k+1)*ones(NP,1);
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGL1(:,k) = lG(XK,XKM,Cova);
            %h_hat, conditional density 
            lGL2(:,k) = 0;
            
            % the integral of L
            % if there is any way to replace the for loop
            for m = 0:nl-1
                lGL3(:,k) = lGL3(:,k) + hl*diag((-A*X_Bri(:,:,m+1)')' * Cova_Inv*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
            end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);
        end
        GL0 = exp(lGL(:,k) - max(lGL(:,k)));
        lGL_star(1,k)= log(sum(GL0)) + max(lGL(:,k));
        GLL = GL0 / sum(GL0);
     
        I = resampleSystematic( GLL);
        X_est(:,1,1:k) = X_est(I, 1,1:k);
        X_est(:,2,1:k) = X_est(I, 2,1:k); 
                
    end
    
    z = T * log(1/NP) + sum(lGL_star);
end


% the auxillary diffusion is dx_t = sqrtm(Cov)*dW_t with fixed starting point, 
% then in a unit interval[t-1,t], x_t ~ N(x_{t-1}, Cov),we use the conditional transition density
% of the auxillary diffusion to sample the missing observations.

%Euler discretization of the diffusion bridge
function X_bridge = em_b(X_start, X_end, L, N, a11,a12,a21,a22,sigma1,sigma2,rho)
    hh = 2^-L;
    nh = 2^L;
    X_bridge = zeros(N,2,nh);
    X_bridge(:,:,1) = X_start;
    R = chol(eye(2) * hh);
    dW = zeros(N,2,nh);
    for m = 1:nh-1
        dW(:,:,m) = (R*randn(N,2)')';
    end
    SIG = sqrtm([sigma1^2,rho*sigma1*sigma2;rho*sigma1*sigma2,sigma2^2]);
    for ii = 1:nh-1 
        X_bridge(:,:,ii+1) = X_bridge(:,:,ii) - ([a11,a12;a21,a22]*X_bridge(:,:,ii)')'*hh + (X_end-X_bridge(:,:,ii))/(1-(ii-1)*hh)*hh + (SIG*dW(:,:,ii)')';
    end
end

%delta pf
function [lwf, lwc, cz] = c_pf_db(L, NP, T, X,A1,A2,A3,A4,SIG1,SIG2,rho)
    hlf = 2^(-L);
    nlf = 2^L;
    hlc = 2^(-(L-1));
    nlc = 2^(L-1);
    A  = [A1,A2;A3,A4];
    
    Xf_est = zeros(NP,2,T+1);
    Xf_est(:,1,1) = X(1,1) * ones(NP,1);
    Xf_est(:,2,1) = X(2,1) * ones(NP,1);

    Xc_est = zeros(NP,2,T+1);
    Xc_est(:,1,1) = X(1,1) * ones(NP,1);
    Xc_est(:,2,1) = X(2,1) * ones(NP,1);

    Cova = [SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2];
    Cova_Inv = Cova^(-1);
    lGLf1 = zeros(NP,T);
    lGLf2 = zeros(NP,T);
    lGLf3 = zeros(NP,T);
    lGLf = zeros(NP,T);

    lGLc1 = zeros(NP,T);
    lGLc2 = zeros(NP,T);
    lGLc3 = zeros(NP,T);
    lGLc = zeros(NP,T);

    lGLJ = zeros(NP,T);
    lGL_star = zeros(1,T);
    lwf = 0;
    lwc = 0;
  
    
     for k = 1:T
        XKf = zeros(NP,2);
        XKMf = zeros(NP,2);
        XKc = zeros(NP,2);
        XKMc = zeros(NP,2);
        
        XKMf(:,1) = Xf_est(:,1,k);
        XKMf(:,2) = Xf_est(:,2,k);
        XKMc(:,1) = Xc_est(:,1,k);
        XKMc(:,2) = Xc_est(:,2,k);
        %update x_{k+1}
        %the first observation is missed
        %simulate x^1_{k+1}conditional on X^2_{k+1} and X_k
        if X(1,k+1) == 0 && X(2,k+1) ~= 0
            Xf_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xf_est(:,1,k+1) = sqrt((1-rho^2))*SIG1*(0.1)*randn(NP,1) + Xf_est(:,1,k) + rho*SIG1/SIG2* (Xf_est(:,2,k+1)-Xf_est(:,2,k));
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xc_est(:,1,k+1) = sqrt((1-rho^2))*SIG1*(0.1)*randn(NP,1) + Xc_est(:,1,k) + rho*SIG1/SIG2* (Xc_est(:,2,k+1)-Xc_est(:,2,k));
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);


            [Xf_Bri,Xc_Bri] = cou_em_b(XKMf, XKf, XKMc,XKc, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion Xf_tk+1 ~ N(Xf_tk,Sigma*Sigma'*1)
            lGLf1(:,k) = lG(XKf,XKMf,Cova);
            %h_hat, conditional density 
            lGLf2(:,k) = -lG(Xf_est(:,1,k+1),Xf_est(:,1,k) + rho*SIG1/SIG2*(Xf_est(:,2,k+1)-Xf_est(:,2,k)),(1-rho^2)*SIG1^2*0.01);            
            % the integral of L
            for m1 = 0:nlf-1
                lGLf3(:,k) = lGLf3(:,k) + diag(hlf*(-A*Xf_Bri(:,:,m1+1)')' * Cova_Inv*(XKf-Xf_Bri(:,:,m1+1))'/(1-m1*hlf));
            end
            
            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

            lGLc1(:,k) = lG(XKc,XKMc,Cova);
            %h_hat, conditional density 
            lGLc2(:,k) = -lG(Xc_est(:,1,k+1),Xc_est(:,1,k) + rho*SIG1/SIG2*(Xc_est(:,2,k+1)-Xc_est(:,2,k)),(1-rho^2)*SIG1^2*0.01);            
            % the integral of L
            for m2 = 0:nlc-1
                lGLc3(:,k) = lGLc3(:,k) + hlc*diag((-A*Xc_Bri(:,:,m2+1)')' * Cova_Inv*(XKc-Xc_Bri(:,:,m2+1))'/(1-m2*hlc));
            end
            
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
            
            %lGLJ(:,k) = max(lGLf(:,k),lGLc(:,k));
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
        %the second observation is missed
        %simulate x^2_{k+1}conditional on X^1_{k+1} and X_k
        elseif X(2,k+1) == 0 && X(1,k+1) ~= 0
            Xf_est(:,1,k+1) = X(1,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xf_est(:,2,k+1) = sqrt((1-rho^2))*SIG2*(0.1)*randn(NP,1) + Xf_est(:,2,k) + rho*SIG2/SIG1* (Xf_est(:,1,k+1)-Xf_est(:,1,k));
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,1,k+1) = X(1,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xc_est(:,2,k+1) = sqrt((1-rho^2))*SIG2*(0.1)*randn(NP,1) + Xc_est(:,2,k) + rho*SIG2/SIG1* (Xc_est(:,1,k+1)-Xc_est(:,1,k));
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);


            [Xf_Bri,Xc_Bri] = cou_em_b(XKMf, XKf, XKMc, XKc, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGLf1(:,k) = lG(XKf,XKMf,Cova);
            %h_hat, conditional density 
            lGLf2(:,k) = -lG(Xf_est(:,2,k+1),Xf_est(:,2,k) + rho*SIG2/SIG1*(Xf_est(:,1,k+1)-Xf_est(:,1,k)),(1-rho^2)*SIG2^2*0.01);            
            % the integral of L
            % if there is any way to replace the for loop
            %
            for m1 = 0:nlf-1
                lGLf3(:,k) = lGLf3(:,k) + hlf*diag((-A*Xf_Bri(:,:,m1+1)')' * Cova_Inv*(XKf-Xf_Bri(:,:,m1+1))'/(1-m1*hlf));
            end
            
            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

            lGLc1(:,k) = lG(XKc,XKMc,Cova);
            %h_hat, conditional density 
            lGLc2(:,k) = -lG(Xc_est(:,2,k+1),Xc_est(:,2,k) + rho*SIG2/SIG1*(Xc_est(:,1,k+1)-Xc_est(:,1,k)),(1-rho^2)*SIG2^2*0.01);            
            % the integral of L
            
            for m2 = 0:nlc-1
                lGLc3(:,k) = lGLc3(:,k) + hlc*diag((-A*Xc_Bri(:,:,m2+1)')' * Cova_Inv*(XKc-Xc_Bri(:,:,m2+1))'/(1-m2*hlc));
            end
           
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
            
            %lGLJ(:,k) = max(lGLf(:,k), lGLc(:,k));
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
        
        %no missing components
        elseif  X(2,k+1) ~= 0 && X(1,k+1) ~= 0
            Xf_est(:,1,k+1) = X(1,k+1)*ones(NP,1);
            Xf_est(:,2,k+1) = X(2,k+1)*ones(NP,1);
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,1,k+1) = X(1,k+1)*ones(NP,1);
            Xc_est(:,2,k+1) = X(2,k+1)*ones(NP,1);
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);

            [Xf_Bri,Xc_Bri] = cou_em_b(XKMf, XKf, XKMc, XKc, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGLf1(:,k) = lG(XKf,XKMf,Cova);
            %h_hat, conditional density 
            lGLf2(:,k) = 0;
            % the integral of L
            
            for m1 = 0:nlf-1
                lGLf3(:,k) = lGLf3(:,k) + hlf*diag((-A*Xf_Bri(:,:,m1+1)')' * Cova_Inv*(XKf-Xf_Bri(:,:,m1+1))'/(1-m1*hlf));
            end
          
            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

            lGLc1(:,k) = lG(XKc,XKMc,Cova);
            %h_hat, conditional density 
            lGLc2(:,k) = 0;
            % the integral of L
            
            for m2 = 0:nlc-1
                lGLc3(:,k) = lGLc3(:,k) + hlc*diag((-A*Xc_Bri(:,:,m2+1)')' * Cova_Inv*(XKc-Xc_Bri(:,:,m2+1))'/(1-m2*hlc));
            end
            
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));

        end
        GL0 = exp(lGLJ(:,k) - max(lGLJ(:,k)));
        lGL_star(1,k)= log(sum(GL0)) + max(lGLJ(:,k));
        lwf = lwf + mean((lGLf(:,k) - lGLJ(:,k)));
        lwc = lwc + mean((lGLc(:,k) - lGLJ(:,k)));

        
        GLL = GL0 / sum(GL0);
        I = resampleSystematic( GLL);
        Xf_est(:,1,1:k) = Xf_est(I, 1,1:k);
        Xf_est(:,2,1:k) = Xf_est(I, 2,1:k);
        
        Xc_est(:,1,1:k) = Xc_est(I, 1,1:k);
        Xc_est(:,2,1:k) = Xc_est(I, 2,1:k);
  
     end

    cz = T * log(1/NP) + sum(lGL_star);

end
%coupled bridge
function [Xf_bridge,Xc_bridge] = cou_em_b(Xf_start, Xf_end, Xc_start,Xc_end, L, NP, a11,a12,a21,a22,sigma1,sigma2,rho)
    hhf = 2^-L;
    nhf = 2^L;
    hhc = 2^(-(L-1));
    nhc = 2^(L-1);
    Xf_bridge = zeros(NP,2,nhf);
    Xf_bridge(:,:,1) = Xf_start;
    Xc_bridge = zeros(NP,2,nhc);
    Xc_bridge(:,:,1) = Xc_start;

  
    R = chol(eye(2) * hhf);
    dWf = zeros(NP,2,nhf-1);
    dWc = zeros(NP,2,nhc-1);
    for m = 1:nhf
        dWf(:,:,m) = (R*randn(NP,2)')';
    end
    SIG = sqrtm([sigma1^2,rho*sigma1*sigma2;rho*sigma1*sigma2,sigma2^2]);
    for ii = 1:nhc-1 
        dWc(:,:,ii) = zeros(NP,2);
        for jj = 1:2
            Xf_bridge(:,:,2*(ii-1)+jj+1) = Xf_bridge(:,:,2*(ii-1)+jj) - ([a11,a12;a21,a22]*Xf_bridge(:,:,2*(ii-1)+jj)')'*hhf + (Xf_end-Xf_bridge(:,:,2*(ii-1)+jj))/(1-(2*(ii-1)+(jj-1))*hhf)*hhf + (SIG*dWf(:,:,2*(ii-1)+jj)')';
            dWc(:,:,ii) = dWc(:,:,ii) + dWf(:,:,2*(ii-1)+jj);
        end
        Xc_bridge(:,:,ii+1) = Xc_bridge(:,:,ii) - ([a11,a12;a21,a22]*Xc_bridge(:,:,ii)')'*hhc + (Xc_end-Xc_bridge(:,:,ii))/(1-(ii-1)*hhc)*hhc + (SIG*dWc(:,:,ii)')';
    end
    Xf_bridge(:,:,nhf) = Xf_bridge(:,:,nhf-1) - ([a11,a12;a21,a22]*Xf_bridge(:,:,nhf-1)')'*hhf + (Xf_end-Xf_bridge(:,:,nhf-1))/(1-((nhf-2))*hhf)*hhf + (SIG*dWf(:,:,nhf)')';
end

function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end


function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    %log_prior = lG(Theta(1),0,1)+lG(Theta(2),0,1)+lG(Theta(3),0,1)+lG(Theta(4),0,1);    
    log_prior = lG(Theta(1),0,1);
    lpos_p = log_lik + log_prior;
    
end

function  indx  = resampleSystematic(w)

N = length(w);
Q = cumsum(w);
indx = zeros(N,1);
T = linspace(0,1-1/N,N) + rand(1)/N;
T(N+1) = 1;

i=1;
j=1;

while (i<=N)
    if (T(i)<Q(j))
        indx(i)=j;
        i=i+1;
    else
        j=j+1;        
    end
end
end



function [In1, In2] = coupled_resampling(N, w1, w2)
%% Coupled resampling

    alphan = sum(min(w1,w2));
    if alphan == 1 %this is an error. Alpha must be less than 1.
      display(w1)
      display(w2)
      error('It seems the wieghts of all particles are zero except one.\n')
    end


    r = rand;
    if r < alphan
        prob = min(w1, w2)/alphan;
        In1 = randsample(N,N,true,prob);
        In2 = In1;
    else
        prob = (w1 - min(w1, w2))/(1-alphan);
        In1 = randsample(N,N,true,prob);
        prob = (w2 - min(w1, w2))/(1-alphan);
        In2 = randsample(N,N,true,prob);
    end

end
