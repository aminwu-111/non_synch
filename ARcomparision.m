% This code is used to generate compare pmcmc convergence 
% resultes for different acceptance rates(07 and 23) Figure 4
warning('off', 'all');
clear
close all
clc
format long
data = [
    0.52, 0.15, 0.33;
    0.47, 0.21, 0.32;
    0.43, 0.27, 0.30;
    0.41, 0.31, 0.29;
    0.38, 0.35, 0.26;
    0.40, 0.36, 0.27;
    0.36, 0.37, 0.37;
    0.35, 0.38, 0.30;
    0.35, 0.38, 0.27;
    0.36, 0.39, 0.28;
    0.37, 0.39, 0.24;
    0.39, 0.39, 0.23;
    0.38, 0.39, 0.23;
    0.37, 0.39, 0.29;
    0.38, 0.39, 0.22;
    0.38, 0.39, 0.32;
    0.34, 0.38, 0.28;
];

X = data(:, 1:2);


periods = {
    '1998a';
    '1998b';
    '1999a';
    '1999b';
    '2000a';
    '2000b';
    '2001a';
    '2001b';
    '2002a';
    '2002b';
    '2003a';
    '2003b';
    '2004a';
    '2004b';
    '2005a';
    '2005b';
    '2006a';
    '2006b';
    '2007a';
};

numnumericPeriods = zeros(length(periods), 1);

% 转换为数值
for i = 1:length(periods)
    year = str2double(periods{i}(1:4));
    if periods{i}(5) == 'a'
        numericPeriods(i) = year - 1998;
    elseif periods{i}(5) == 'b'
        numericPeriods(i) = year - 1998 + 0.5;
    end
end

T_obs = numericPeriods';
T_obs = T_obs(1:end-1);
T = length(T_obs);

X_miss = X;

I1 = randi([0,1],1,T);
I2 = randi([0,1],1,T);
for j = 1:T-2
    if I1(j) == 0 && I2(j) == 0
        index = randi([1,2]);
        X_miss(j+1, index) = 0;
    else
        X_miss(j+1, 1) = X_miss(j+1, 1) * I1(j);
        X_miss(j+1, 2) = X_miss(j+1, 2) * I2(j);
    end
end
X = X_miss;

warning('off', 'all');



NP = 40;
%NP = 30 %For the  var(losposterior) around 1.8^2
Lmin = 5;
LP = 5;

% discretization size
hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

Nl = 5000;
Aln = zeros(LP - Lmin + 1, 2);
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

Lpos = zeros(Nl,1);
Theta_A = [-5,-5,-6,-5];
tic;

Theta_A_p = Theta_A;
Theta_SIG_p = [exp(Theta_A_p(1:4))];
Z = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),0.08,0.17,-0.46);
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace{1,1}(1,:) = Theta_A_p;
Lpos(1) = l_pos_Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;

Sigma_A1 = 0.55*diag([4,3,3,4]); %23 acceptance rate set
%Sigma_A1 = 0.98*diag([4,4,3.5,4]); % 07acceptance rate


for iter = 1:Nl(1)
 
    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        %disp(['var_lpos = ', num2str(var(Lpos(1:iter)))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)), ', ', num2str(Theta_A_p(3)), ', ', num2str(Theta_A_p(4)), ']']);
        %disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1f = Theta_A_p(1:4);
    Theta_A_prime_1f = mvnrnd(Theta_A_prime_1f, Sigma_A1*Sigma_A1');
    Theta_A_prime_1 = [Theta_A_prime_1f];
    Theta_SIG_prime_1 = [exp(Theta_A_prime_1(1:4))];
   
    Z_prime_1 = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_prime_1(1),Theta_SIG_prime_1(2),Theta_SIG_prime_1(3),Theta_SIG_prime_1(4),0.08,0.17,-0.46);
    lZ_prime_1 = Z_prime_1;
    l_pos_Theta_A_prime_1 = l_posterior(Theta_A_prime_1, lZ_prime_1);
 
    alpha_U1 = min(0, l_pos_Theta_A_prime_1 - l_pos_Theta_A_p);
    U1 = log(rand);
    
    if U1 < alpha_U1
        Theta_A_p = Theta_A_prime_1;
        Theta_SIG_p = Theta_SIG_prime_1;
        lZ = lZ_prime_1;
        l_pos_Theta_A_p = l_pos_Theta_A_prime_1;
        Theta_trace{1, 1}(iter,:) = Theta_A_prime_1; 
        Lpos(iter) = l_pos_Theta_A_prime_1;
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace{1, 1}(iter,:) = Theta_A_p; 
        lZ = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),0.08,0.17,-0.46);
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        Lpos(iter) =  l_pos_Theta_A_p2;
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 
end

Aln(1,1) = N_count_1 / Nl(1);

Theta_iters = Theta_trace{1,1};
burnin = 1;
niter = 5000;
desired_height = 0.12;
figure_distance = 400;
f = figure;
f.Position = f.Position+[0 -figure_distance 0 figure_distance];

ax = subplot(4,1,1);
plot(burnin:3:niter,Theta_iters(burnin:3:end,1), 'r-',LineWidth=1);
title('log(\alpha)');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(4,1,2);
plot(burnin:3:niter,Theta_iters(burnin:3:end,2), 'r-',LineWidth=1);
title('log(\beta)');
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');


ax = subplot(4,1,3);
plot(burnin:3:niter,Theta_iters(burnin:3:end,3), 'r-',LineWidth=1);
title('log(\zeta)')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');

ax = subplot(4,1,4);
plot(burnin:3:niter,Theta_iters(burnin:3:end,4), 'r-',LineWidth=1);
title('log(\gamma)')
current_height = ax.Position(4);
remaning_height = desired_height - current_height;
ax.Position = ax.Position + [0 -remaning_height/2 0 remaning_height/2];
ylabel('convergence value');
xlabel('iterations');



%particle filter using diffusion bridge
function z = PF_DB(L, NP, T, T_obs, X, a1,a2,a3,a4,s1,s2,rho)

    hl = 2^(-L);
    X_est = zeros(NP,2,T);
    X_est(:,1,1) = X(1,1) * ones(NP,1);
    X_est(:,2,1) = X(1,2) * ones(NP,1);
    lGL1 = zeros(NP,T-1);
    lGL2 = zeros(NP,T-1);
    lGL3 = zeros(NP,T-1);
    lGL = zeros(NP,T-1);
    lGL_star = zeros(1,T-1);

    for k = 1:T - 2

        XK = zeros(NP,2);
        XKM = zeros(NP,2);
        XKM(:,1) = X_est(:,1,k);
        XKM(:,2) = X_est(:,2,k);
        steps = T_obs(k+1)- T_obs(k);
        nl = ceil(steps/hl);
       
       
        if X(k+1,2) ~= 0 && X(k+1,1) == 0

            X_est(:,2,k+1) = X(k+1,2) * ones(NP,1);
             [samp, log_den] = h_log_normal(NP, steps,a1,a2,a3,a4,s1,s2, rho, XKM(:,1), XKM(:,2), -1, X(k+1,2));
            
            for n = 1:NP 
                X_est(n,1,k+1) = samp(1,n);
                lGL2(n,k) = log_den(n);
            end
           

            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
    
            [den_tilde,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2, a3,a4, s1, s2,rho, XKM(:,1), XKM(:,2), XK(:,1), XK(:,2),XKM(:,1), XKM(:,2));
            lGL1(:,k) = den_tilde';

            [X_Bri, deri_logx] = em_b_linear(XKM, XK, T_obs(k),T_obs(k+1), L, NP, a1,a2,a3,a4,s1,s2,rho);
           
            Rc = X_Bri(:,1,1);
            Fc = X_Bri(:,2,1);
            Rb = X_Bri(:,1,1);
            Fb = X_Bri(:,2,1);
            Re = XK(:,1);
            Fe = XK(:,2);
            % the integral of L
            for m = 0:nl-2 
                dif_mu = zeros(NP,2);
                tc = m*hl;
                t1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_mu(n,:) = [a2*Rc(n)*(Fb(n)- Fc(n) + tc/t1*(Fe(n)-Fb(n))), a3*Fc(n)*(Rc(n)- Rb(n)+ tc/t1*(Rb(n)-Re(n)))];
                    dri_f = deri_logx(n,:,m+1);
                    lGL3(n,k) = lGL3(n,k) + hl*(dif_mu(n,:)*dri_f');
                end

                if m ~= nl-2
                    Rc = X_Bri(:,1,m+2);
                    Fc = X_Bri(:,2,m+2);
                end
            end

              tce = (nl-1)*hl;
              hle = t1 - tce;
              Rce = X_Bri(:,1,nl);
              Fce = X_Bri(:,2,nl);
              dif_mue = zeros(NP,2);
              
              for n = 1:NP
                  dif_mue(n,:) = [a2*Rce(n)*(Fb(n)- Fce(n) + tce/t1*(Fe(n)-Fb(n))), a3*Fce(n)*(Rce(n)- Rb(n)+ tce/t1*(Rb(n)-Re(n)))];
                  dri_fe = deri_logx(n,:,nl);
                  lGL3(n,k) = lGL3(n,k) + hle*(dif_mue(n,:)*dri_fe');
              end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);

             elseif X(k+1,2) == 0 && X(k+1,1) ~= 0
    
            X_est(:,1,k+1) = X(k+1,1) * ones(NP,1);
            
            [samp, log_den] = h_log_normal(NP, steps, a1,a2,a3,a4,s1,s2, rho, XKM(:,1), XKM(:,2), X(k+1,1), -1);
            for n = 1:NP 
                X_est(n,2,k+1) = samp(2,n);
                lGL2(n,k) = log_den(n);
            end
            
          
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);


            %calculate the pdf from f_tilde 
            [den_tilde,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2, a3,a4, s1, s2,rho, XKM(:,1), XKM(:,2), XK(:,1), XK(:,2),XKM(:,1), XKM(:,2));
            lGL1(:,k) = den_tilde';


            [X_Bri, deri_logx] = em_b_linear(XKM, XK, T_obs(k),T_obs(k+1), L, NP, a1,a2,a3,a4,s1,s2,rho);
           
            Rc = X_Bri(:,1,1);
            Fc = X_Bri(:,2,1);
            Rb = X_Bri(:,1,1);
            Fb = X_Bri(:,2,1);
            Re = XK(:,1);
            Fe = XK(:,2);
            % the integral of L
            for m = 0:nl-2 
                dif_mu = zeros(NP,2);
                tc = m*hl;
                t1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_mu(n,:) = [a2*Rc(n)*(Fb(n)- Fc(n) + tc/t1*(Fe(n)-Fb(n))), a3*Fc(n)*(Rc(n)- Rb(n)+ tc/t1*(Rb(n)-Re(n)))];
                    dri_f = deri_logx(n,:,m+1);
                    lGL3(n,k) = lGL3(n,k) + hl*(dif_mu(n,:)*dri_f');
                   
                end

                if m ~= nl-2
                    Rc = X_Bri(:,1,m+2);
                    Fc = X_Bri(:,2,m+2);
                end
            end

              tce = (nl-1)*hl;
              hle = t1 - tce;
              Rce = X_Bri(:,1,nl);
              Fce = X_Bri(:,2,nl);
              dif_mue = zeros(NP,2);
              
              for n = 1:NP
                  dif_mue(n,:) = [a2*Rce(n)*(Fb(n)- Fce(n) + tce/t1*(Fe(n)-Fb(n))), a3*Fce(n)*(Rce(n)- Rb(n)+ tce/t1*(Rb(n)-Re(n)))];
                  dri_fe = deri_logx(n,:,nl);
                  lGL3(n,k) = lGL3(n,k) + hle*(dif_mue(n,:)*dri_fe');
              end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);

    elseif X(k+1,2) ~= 0 && X(k+1,1) ~= 0

            X_est(:,1,k+1) = X(k+1,1)*ones(NP,1);
            X_est(:,2,k+1) = X(k+1,2)*ones(NP,1);
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
            [log_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,s1,s2,rho, XKM(:,1), XKM(:,2), XK(:,1), XK(:,2),XKM(:,1), XKM(:,2));
            
            for n = 1:NP 
                lGL1(n,k) = log_density(n);
            end

            lGL2(:,k) = 0;

            [X_Bri, deri_logx] = em_b_linear(XKM, XK, T_obs(k),T_obs(k+1), L, NP, a1,a2,a3,a4,s1,s2,rho);
               
            Rc = X_Bri(:,1,1);
            Fc = X_Bri(:,2,1);
            Rb = X_Bri(:,1,1);
            Fb = X_Bri(:,2,1);
            Re = XK(:,1);
            Fe = XK(:,2);
            % the integral of L
            for m = 0:nl-2 
                dif_mu = zeros(NP,2);
                tc = m*hl;
                t1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_mu(n,:) = [a2*Rc(n)*(Fb(n)- Fc(n) + tc/t1*(Fe(n)-Fb(n))), a3*Fc(n)*(Rc(n)- Rb(n)+ tc/t1*(Rb(n)-Re(n)))];
                    dri_f = deri_logx(n,:,m+1);
                    lGL3(n,k) = lGL3(n,k) + hl*(dif_mu(n,:)*dri_f');
                end

                if m ~= nl-2
                    Rc = X_Bri(:,1,m+2);
                    Fc = X_Bri(:,2,m+2);
                end
            end

              tce = (nl-1)*hl;
              hle = t1 - tce;
              Rce = X_Bri(:,1,nl);
              Fce = X_Bri(:,2,nl);
              dif_mue = zeros(NP,2);
              
              for n = 1:NP
                  dif_mue(n,:) = [a2*Rce(n)*(Fb(n)- Fce(n) + tce/t1*(Fe(n)-Fb(n))), a3*Fce(n)*(Rce(n)- Rb(n)+ tce/t1*(Rb(n)-Re(n)))];
                  dri_fe = deri_logx(n,:,nl);
                  lGL3(n,k) = lGL3(n,k) + hle*(dif_mue(n,:)*dri_fe');
              end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);
        end

        GL0 = exp(lGL(:,k) - max(lGL(:,k)));
        lGL_star(1,k)= log(sum(GL0)) + max(lGL(:,k));
        GLL = GL0 / sum(GL0);
        if isnan(sum(GLL)) 
            disp('ANNOYING NAN ERROR! GLL');
            z = -inf;
            return
        end
        if  sum(GLL) == 0
            disp(' GLL = 0');
            z = -inf;
            return
        end
        
        I = resampleSystematic( GLL);
        X_est(:,1,1:k) = X_est(I, 1,1:k);
        X_est(:,2,1:k) = X_est(I, 2,1:k); 
            
    end
    
    z = (T - 1) * log(1/NP) + sum(lGL_star);
end

function [samples, log_density] = h_log_normal(N, tdiff, a11,a12, a21,a22, sig1, sig2, rho, R0, F0, Re, Fe)
    samples = zeros(2,N);
    log_density = zeros(1,N);
    if any(Re == -1)
        for n = 1:N
            sigma = 0.1;
            mu = log(R0(n)) - sigma^2/2;
            X = lognrnd(mu, sigma);
            samples(1,n) = X;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
            samples(2,n) = Fe;
        end
    end

    if any(Fe == -1)
        for n = 1:N
            sigma = 0.1;
            mu = log(F0(n)) - sigma^2/2;
            X = lognrnd(mu, sigma);
            samples(2,n) = X;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
            samples(1,n) = Re;
        end
    end
end

function [log_density, grad_log_density] = linear_tran_t_aux(N, t0, t1, a11,a12, a21,a22, sig1, sig2, rho, R0, F0, Re, Fe, Rc, Fc)
    
    grad_log_density = zeros(2,N);
    log_density = zeros(1,N);
    tdiff = t1 - t0;

    for n = 1:N
            
            p_1 = (log(Re(n)/Rc(n)) - (a11 - a12*F0(n) - sig1^2/2)*tdiff - a12*(F0(n) - Fe(n))*(t1^2 - t0^2)/(2*t1))/(sig1*sqrt(tdiff));
            p_2 = (log(Fe(n)/Fc(n)) - (a21*R0(n) - a22 - sig2^2/2)*tdiff - a21*(Re(n) - R0(n))*(t1^2 - t0^2)/(2*t1))/(sig2*sqrt(tdiff));
            SIG = [1,rho;rho,1];
            SIG_inv = inv(SIG); 
            log_density(n) = lG([p_1,p_2],[0,0],SIG) - log(sig1*sig2*tdiff*Re(n)*Fe(n));

            grad_log_density(1,n) = 1/(sig1*sqrt(tdiff)) * 1/Rc(n) * (SIG_inv(1,1)*p_1 + SIG_inv(1,2)*p_2);
            grad_log_density(2,n) = 1/(sig2*sqrt(tdiff)) * 1/Fc(n) * (SIG_inv(2,1)*p_1 + SIG_inv(2,2)*p_2);
    end
end



function [X_bridge, deri_x] = em_b_linear(X_start, X_end, t0, t1, L, N, a11,a12,a21,a22,sig1,sig2,rho)
    
    hh = 2^(-L);
    dis = t1 - t0;
    nh = ceil(dis/hh);
    Re = X_end(:,1);
    Fe = X_end(:,2);
    Rb = X_start(:,1);
    Fb = X_start(:,2);

    X_bridge = zeros(N,2,nh);
    X_bridge(:,:,1) = X_start;
    deri_x = zeros(N,2,nh);

    lg_X_bridge = zeros(N,2,nh);
    lg_X_bridge(:,:,1) = log(X_start);

   
    for ii = 1:nh-1

        %generate the correlated brownian motion
        Z1 = randn(N, 1);
        Z2 = randn(N, 1);
        Z2_corr = rho * Z1 + sqrt(1 - rho^2) * Z2;
        Z1_corr = Z1;
        dW1 = sqrt(hh) * Z1_corr;
        dW2 = sqrt(hh) * Z2_corr;
        dW = [dW1,dW2];
       
        % Take log(Rt) log(Ft) and then take exp

        Rs = X_bridge(:,1,ii);
        Fs = X_bridge(:,2,ii);    

        [~,grad_densities] = linear_tran_t_aux(N, (ii-1)*hh,t1-t0, a11,a12, a21,a22, sig1, sig2,rho, Rb, Fb, Re, Fe, Rs, Fs);
        deri_x(:,:,ii) = grad_densities';

        for n = 1:N
            dri_t1 = [a11 - a12*Fs(n); a21*Rs(n) - a22]*hh;
            dri_t2 = [sig1^2*Rs(n),rho*sig1*sig2*Fs(n);rho*sig1*sig2*Rs(n),sig2^2*Fs(n)]*deri_x(n,:,ii)'*hh;
            dri_t3 = [-sig1^2/2;-sig2^2/2]*hh;
            dif_t = [sig1,0;0,sig2]*dW(n,:)';
            lg_X_bridge(n,:,ii+1) = lg_X_bridge(n,:,ii) + (dri_t1 + dri_t2 + dri_t3 + dif_t)';   
            X_bridge(n,:,ii+1) = exp(lg_X_bridge(n,:,ii+1));
        end

    end
end


function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end


function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior =  lG(Theta(1),-5,3)+lG(Theta(2),-5,3)+lG(Theta(3),-6,3)+lG(Theta(4),-5,3);
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
