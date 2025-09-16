%PMCMC with EM
close all;
clear;
clc;
format long

T = 65;       
A = [0.8,0.2;-0.3,0.8];
COV = [1,0.5; 0.5,1];
Sigma = sqrtm(COV);
rho = 0.5;
load('X_18.mat');

x0 = X(:,1);

NP = 100;
LP = 4;
niter = 10000;
Theta_A = [0.9,0.2,-0.3,0.6];

Theta_trace_A = zeros(niter,4);
Lpos = zeros(niter,1);

Theta_A_p = Theta_A;
Sigma_A = 0.6*diag([0.8,0.6,0.6,0.8]);

Z = PF_EM(LP, NP, T, X, x0,Theta_A_p(1),Theta_A_p(2),Theta_A_p(3),Theta_A_p(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Lpos(1) = l_pos_Theta_A_p;

N_count = 0;
N_count_last = 0;
tic
for iter = 1:niter
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

    Z_prime = PF_EM(LP, NP, T, X, x0, ...
            Theta_A_prime(1),Theta_A_prime(2),Theta_A_prime(3),Theta_A_prime(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
    lZ_prime = Z_prime;
    l_pos_Theta_A_prime = l_posterior(Theta_A_prime, lZ_prime);
    
    
    alpha = min(0, l_pos_Theta_A_prime - l_pos_Theta_A_p);
    U = log(rand);
    if U < alpha
        Theta_A_p = Theta_A_prime;
        lZ = lZ_prime;
        l_pos_Theta_A_p = l_pos_Theta_A_prime;
        Theta_trace_A(iter, :) = Theta_A_prime; 
        N_count = N_count + 1;
        Lpos(iter) = l_pos_Theta_A_prime;
    else
        Theta_trace_A(iter, :) = Theta_A_p; 
        lZ = PF_EM(LP, NP, T, X, x0, Theta_A_p(1),Theta_A_p(2),Theta_A_p(3),Theta_A_p(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
        l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
        Lpos(iter) = l_pos_Theta_A_p;
    end 
end
toc
AR = N_count / (niter);


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
        Cov = [SIG1^2,rho*SIG1*SIG2;rho*SIG1*SIG2,SIG2^2]* hl;

        %conditional likelihood X2|X1_est
        %the first observation is missed
        if X(1,k+1) == 0
            %generate x_est(:,1)
            x_est(:,2) = X(2,k+1) * ones(N,1);
            x_est(:,1) = sqrt((1-rho^2) * hl)*SIG1*randn(N,1) + mu(:,1,k+1) + rho*SIG1/SIG2* (x_est(:,2)-mu(:,2,k+1));
            lGL= lG(x_est(:,2), mu(:,2,k+1)+ rho*SIG2/SIG1*(x_est(:,1) - mu(:,1,k+1)), (1-rho^2)*SIG2^2*hl);
            mu(:,2,k+1) = X(2,k+1) .* ones(N,1);
            mu(:,1,k+1) = x_est(:,1);
        %the second observation is missed
        elseif X(2,k+1) == 0
            %generate x_est(:,2)
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
    %z = exp(log(GLL) + T * log(1/N) + sum(lGL_star));  
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


function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * inv(Cov) * (y-x)') ;
end


function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior = lG(Theta(1),0,1)+lG(Theta(2),0,1)+lG(Theta(3),0,1)+lG(Theta(4),0,1);
    
    %log_prior = lG(Theta(1),0,1);
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

