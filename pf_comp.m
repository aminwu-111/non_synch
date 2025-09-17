% This code is used to compare the variance of the likelihood for 
% particle filter using  EM discretization and diffusion bridge 

close all
clear
clc
format long
rng(1)

T = 50;       
L = 10;
delta = 2^(-L); 
nh = 1/delta;
nl = T*2^L; 

% parameters 
A = [0.8,0.2;-0.3,0.8];
x0 = randn(2,1);
R = chol(eye(2) * delta);
dW = cell(nl,1);
for i = 1:nl
    dW{i,1} = R * randn(2,1);
end
X = zeros(2, T+1);
X(:,1) = x0;


%generating data
COV = [1,0.5; 0.5,1];
Sigma = sqrtm(COV);
rho = 0.5;
for k = 1: T
    v = X(:,k);
    
    for i = 1:nh
        v = v-(A*delta)*v + Sigma*dW{(k-1)*nh +i,1};
    end
    X(:,k+1) = v;
end
X(1,2:4:T) = 0;
X(2,3:4:T) = 0;
N_t = 100;

%returned likelihood from the two differnent particles
Li_EM = zeros(100,6);
Li_DB = zeros(100,6);
for i = 2:8
for k = 1:100
    Li_EM(k,i-1) = PF_EM(i,N_t,T,X,x0,A(1,1),A(1,2),A(2,1),A(2,2),1,1,0.5);
    Li_DB(k,i-1) = PF_DB(i,N_t,T, X, A(1,1),A(1,2),A(2,1),A(2,2),1,1,0.5);
end
end

mean_em = mean(Li_EM,1);
mean_db = mean(Li_DB,1);
var_em = var(Li_EM,1);
var_db = var(Li_DB,1);

figure
plot(2:8, var_em(1:7),'r-', LineWidth=2);
hold on
plot(2:8,var_db(1:7),'b-', LineWidth=2);
hold off
xlabel('levels')
ylabel('likelihood variance')
legend('EM','DB')
title('The variance of the estimated likelihood by using EM and DB')


%particle filter using the EM
function z = PF_EM(L, N, T, X, x0, A1,A2,A3,A4,SIG1,SIG2,rho)
    hl = 2^(-L);
    nl = 2^L;
    nt = T/hl;

    mu = zeros(N,2,T+1);
    x_est = zeros(N,2);
    lGL = zeros(N,1);
    lGL_star = zeros(1,T);
    mu(:,:,1) =  x0' .* ones(N,2);
    for k = 1:T-1
        mu(:,:,k+1)= single_solver(L,N,mu(:,:,k),A1,A2,A3,A4,SIG1,SIG2,rho);
        %%%
        Cov = [SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2]* hl;
        %conditional likelihood X2|X1_est
        %the first observation is missed
        if X(1,k+1) == 0
            
            x_est(:,2) = X(2,k+1) * ones(N,1);
            x_est(:,1) = sqrt((1-rho^2) * hl)*SIG1*randn(N,1) + mu(:,1,k+1) + rho*SIG1/SIG2* (x_est(:,2)-mu(:,2,k+1));
            lGL= lG(x_est(:,2), mu(:,2,k+1)+ rho*SIG2/SIG1*(x_est(:,1) - mu(:,1,k+1)), (1-rho^2)*SIG2^2*hl);
            mu(:,2,k+1) = X(2,k+1) .* ones(N,1);
            mu(:,1,k+1) = x_est(:,1);
        %the second observation is missed
        elseif X(2,k+1) == 0
            
            x_est(:,1) = X(1,k+1) * ones(N,1);
            x_est(:,2) = sqrt((1-rho^2) * hl)*SIG2*randn(N,1) + mu(:,2,k+1) + rho*SIG2/SIG1* (x_est(:,1)-mu(:,1,k+1));
            lGL= lG(x_est(:,1), mu(:,1,k+1)+ rho*SIG1/SIG2*(x_est(:,2) - mu(:,2,k+1)), (1-rho^2)*SIG1^2*hl);
            mu(:,1,k+1) = X(1,k+1) * ones(N,1);
            mu(:,2,k+1) = x_est(:,2);
        else
            x_est(:,1,k+1) = X(1,k+1) * ones(N,1);
            x_est(:,2,k+1) = X(2,k+1) * ones(N,1);
            lGL= lG(x_est(:,:,k+1), mu(:,:,k+1), Cov);
            mu(:,:,k+1) = (X(:,k+1) .* ones(2,N))'; 
        end

        
        GL0 = exp(lGL - max(lGL));
        lGL_star(1,k)= log(sum(GL0)) + max(lGL);
        GLL = GL0 / sum(GL0);  
        I = resampleSystematic( GLL);     
        mu(:,:,2:k+1) = mu(I,:,2:k+1);
        
           
    end      
    z = T * log(1/N) + sum(lGL_star);
    mu(:,:,T+1) = single_solver(L,N,mu(:,:,T),A1,A2,A3,A4,SIG1,SIG2,rho);
    SIG = sqrtm([SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2]);
    mu(:,:,T+1) =  mu(:,:,T+1)-(([A1,A2;A3,A4]*hl) * mu(:,:,T+1)')' + (SIG * chol(hl * eye(2)) * randn(2,N))';
end


function mu = single_solver(L,N,mu_ex,A1,A2,A3,A4,SIG1,SIG2,rho)
    mu = mu_ex;
    h = 2^(-L);
    nl = 2^L;
    A = [A1,A2;A3,A4];
    SIG = sqrtm([SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2]);
    R = chol(h * eye(2));
    dW = cell(nl,1);
    for i = 1:nl-1
        dW{i,1} = R * randn(2,N);
        mu = mu-((A*h) * mu')' + (SIG * dW{i,1})';
    end
end

%particle filter using diffusion bridge
function z = PF_DB(L, NP, T, X,A1,A2,A3,A4,SIG1,SIG2,rho)
    hl = 2^(-L);
    nl = 2^L;
    A  = [A1,A2;A3,A4];
    
    X_est = zeros(NP,2,T+1);
    X_est(:,1,1) = X(1,1) * ones(NP,1);
    X_est(:,2,1) = X(2,1) * ones(NP,1);
    Cova = [SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2];
    lGL1 = zeros(NP,T);
    lGL2 = zeros(NP,T);
    lGL3 = zeros(NP,T);
    lGL = zeros(NP,T);
    lGL_star = zeros(1,T);

    for k = 1:T-1
        XK = zeros(NP,2);
        XKM = zeros(NP,2);
        XKM(:,1) = X_est(:,1,k);
        XKM(:,2) = X_est(:,2,k);
        %the first observation is missed
        %simulate x^1_{k+1}conditional on X^2_{k+1} and X_k
        if X(1,k+1) == 0
            X_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            X_est(:,1,k+1) = sqrt((1-rho^2))*SIG1*randn(NP,1) + X_est(:,1,k) + rho*SIG1/SIG2* (X_est(:,2,k+1)-X_est(:,2,k));
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGL1(:,k) = lG(XK,XKM,Cova);
            %h_hat, conditional density 
            lGL2(:,k) = -lG(X_est(:,1,k+1),X_est(:,1,k) + rho*SIG1/SIG2*(X_est(:,2,k+1)-X_est(:,2,k)),(1-rho^2)*SIG1^2);
            
            % the integral of L
            % if there is any way to replace the for loop
            for m = 0:nl-1
                lGL3(:,k) = lGL3(:,k) + diag(hl*(-A*X_Bri(:,:,m+1)')' * Cova^(-1)*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
            end
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);

        %the second observation is missed
        %simulate x^2_{k+1}conditional on X^1_{k+1} and X_k
        elseif X(2,k+1) == 0
            X_est(:,1,k+1) = X(1,k+1) * ones(NP,1);
            %generate the second comnent from f_hat (browanian motion delta_t = 1)
            X_est(:,2,k+1) = sqrt((1-rho^2))*SIG2*randn(NP,1) + X_est(:,2,k) + rho*SIG2/SIG1* (X_est(:,1,k+1)-X_est(:,1,k));
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGL1(:,k) = lG(XK,XKM,Cova);
            %h_hat, conditional density 
            lGL2(:,k) = -lG(X_est(:,2,k+1),X_est(:,2,k) + rho*SIG2/SIG1*(X_est(:,1,k+1)-X_est(:,1,k)),(1-rho^2)*SIG2^2);
            
            % the integral of L
            for m = 0:nl-1
                lGL3(:,k) = lGL3(:,k) + hl*diag((-A*X_Bri(:,:,m+1)')' * Cova^(-1)*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
            end
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);
        %no missing components
        else
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
                lGL3(:,k) = lGL3(:,k) + hl*diag((-A*X_Bri(:,:,m+1)')' * Cova^(-1)*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
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
    % the last point
    XK(:,:) = (chol(Cova)*randn(NP, 2)')' + XKM;
    X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
    %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
    lGL1(:,T) = lG(XK,XKM,Cova);
    %h_hat, conditional density 
    lGL2(:,T) = -lG(XK,XKM,Cova);
    
    % the integral of L
    % if there is any way to replace the for loop
    for m = 0:nl-1
        lGL3(:,T) = lGL3(:,T) + diag(hl*(A*X_Bri(:,:,m+1)')' * Cova^(-1)*(XK-X_Bri(:,:,m+1))'/(1-m*hl));
    end
    lGL(:,T) = lGL3(:,T)+lGL2(:,T)+lGL1(:,T);
    GL0 = exp(lGL(:,T) - max(lGL(:,T)));
    lGL_star(1,T)= log(sum(GL0)) + max(lGL(:,T));
     
    
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
    for m = 1:nh
        dW(:,:,m) = (R*randn(N,2)')';
    end
    SIG = sqrtm([sigma1^2,rho*sigma1*sigma2;rho*sigma1*sigma2,sigma2^2]);
    for ii = 1:nh-1 
        X_bridge(:,:,ii+1) = X_bridge(:,:,ii) - ([a11,a12;a21,a22]*X_bridge(:,:,ii)')'*hh + (X_end-X_bridge(:,:,ii))/(1-(ii-1)*hh)*hh + (SIG*dW(:,:,ii)')';
    end
end


function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * inv(Cov) * (y-x)') ;
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

