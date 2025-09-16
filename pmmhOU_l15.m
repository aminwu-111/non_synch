

LP = 15;

job_id = getenv('SLURM_JOB_ID');
proc_id = getenv('SLURM_PROCID');
folder_read = '';
folder_write = sprintf('%s%s', job_id, '/');
results_filename = sprintf('%sL_%i_%s_%s.txt', folder_write, LP, job_id,proc_id);


rng_seed = sum(clock)*mod(str2num(job_id),10000)*(str2num(proc_id)+1);
rng(rng_seed);




format long
X = readmatrix(sprintf('%s%s', folder_read,'X.txt'));


T = 65;       % Time

% parameters 
A = [0.8,0.2;-0.3,0.8];
COV = [1,0.5; 0.5,1];
Sigma = sqrtm(COV);
rho = 0.5; 



burnin = 600;
NP = 50; % number of particles
niter =  floor(2*1e-5*2^(2*LP)+ burnin);

%Theta_A = [0.9,0.2,-0.3,0.6];
Theta_A = [0.8, -0.3, 0.3, 0.5];
Theta_trace_A = zeros(niter,4);


Theta_A_p = Theta_A;
Sigma_A = 0.4*diag([0.8,0.6,0.6,0.8]);

Z = PF_DB(LP, NP, T, X, Theta_A_p(1),Theta_A_p(2),Theta_A_p(3),Theta_A_p(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
%lZ = log(sum(Z)+1e-30);
lZ = Z;
Z_col(1) = lZ;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);


N_count = 0;
N_count_last = 0;
num_cac = 0;

for iter = 1:niter

    if mod(iter, 1) == 0
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

    Z_prime = PF_DB(LP, NP, T, X, ...
            Theta_A_prime(1),Theta_A_prime(2),Theta_A_prime(3),Theta_A_prime(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
    %lZ_prime = log(sum(Z_prime)+1e-30);
    lZ_prime = Z_prime;
    Z_col(iter+1) = lZ_prime;
    l_pos_Theta_A_prime = l_posterior(Theta_A_prime, lZ_prime);
    
    
   alpha = min(0, l_pos_Theta_A_prime - l_pos_Theta_A_p);
    U = log(rand);
    if U < alpha
        Theta_A_p = Theta_A_prime;
        %P_Theta_S_p = P_S_prime;
        lZ = lZ_prime;
        l_pos_Theta_A_p = l_pos_Theta_A_prime;
        Theta_trace_A(iter, :) = Theta_A_prime; 
        N_count = N_count + 1;
        
    else
        Theta_trace_A(iter, :) = Theta_A_p; 
        lZ = PF_DB(LP, NP, T, X, Theta_A_p(1),Theta_A_p(2),Theta_A_p(3),Theta_A_p(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
        l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
    end 
end

    AR = N_count / niter;
    

final_theta = mean(Theta_trace_A(burnin:end, :));
writematrix(final_theta, results_filename);




%% functions
%particle filter using diffusion bridge
function z = PF_DB(L, NP, T, X,A1,A2,A3,A4,SIG1,SIG2,rho)
    hl = 2^(-L);
    nl = 2^L;
    %nt = T/hl;
    %Rh = chol(eye(2) * hl);
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
    wa = zeros(NP,1);
    for k = 1:T-1
        XK = zeros(NP,2);
        XKM = zeros(NP,2);
        %update x_{k+1}
        %the first observation is missed
        %simulate x^1_{k+1}conditional on X^2_{k+1} and X_k
        XKM(:,1) = X_est(:,1,k);
        XKM(:,2) = X_est(:,2,k);
        if X(1,k+1) == 0
            X_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            X_est(:,1,k+1) = sqrt((1-rho^2))*SIG1*0.1*randn(NP,1) + X_est(:,1,k) + rho*SIG1/SIG2* (X_est(:,2,k+1)-X_est(:,2,k));
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
            X_est(:,2,k+1) = sqrt((1-rho^2))*SIG2*0.1*randn(NP,1) + X_est(:,2,k) + rho*SIG2/SIG1* (X_est(:,1,k+1)-X_est(:,1,k));
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            X_Bri = em_b(XKM, XK, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGL1(:,k) = lG(XK,XKM,Cova);
            %h_hat, conditional density 
            lGL2(:,k) = -lG(X_est(:,2,k+1),X_est(:,2,k) + rho*SIG2/SIG1*(X_est(:,1,k+1)-X_est(:,1,k)),(1-rho^2)*SIG2^2);
            
            % the integral of L
            % if there is any way to replace the for loop
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
     
    %z = exp(log(GLL) + T * log(1/N) + sum(lGL_star));  
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
    %XU = X_start; 
    R = chol(eye(2) * hh);
    dW = zeros(N,2,nh);
    for m = 1:nh
        dW(:,:,m) = (R*randn(N,2)')';
    end
    SIG = sqrtm([sigma1^2,rho*sigma1*sigma2;rho*sigma1*sigma2,sigma2^2]);
    for ii = 1:nh-1 
        X_bridge(:,:,ii+1) = X_bridge(:,:,ii) - ([a11,a12;a21,a22]*X_bridge(:,:,ii)')'*hh + (X_end-X_bridge(:,:,ii))/(1-(ii-1)*hh)*hh + (SIG*dW(:,:,ii)')';
        %X_bridge(:,:,ii+1) = XU;
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

