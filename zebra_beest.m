% This code is used to estimate parameters for the SLV model for the 'beest_zebra' data using the MLPMCMC method
%the model is 
% dR = R * (alpha - beta*F) * dt + sigma_1*R * dW1(t)
% dF = F * (delta*R - gamma) * dt + sigma_2*F * dW2(t)
% dW1 and dW2 are N(0,dt) with correlation rho
warning('off', 'all');
clear
close all
clc
format long
X_unprocessed = readmatrix('beest_zebra.xlsx');
X_unprocessed(:,1) = X_unprocessed(:,1)-X_unprocessed(1,1);

T_obs = X_unprocessed(:,1);
X = [X_unprocessed(:,3),X_unprocessed(:,2)]/1000;
T_obs = T_obs(1:end-1);

T= length(T_obs);

X_miss = X;

I1 = randi([0,1],1,T);
I2 = randi([0,1],1,T);
for j = 1:T
    if I1(j) == 0 && I2(j) == 0
        index = randi([1,2]);
        X_miss(j+1, index) = 0;
    else
        X_miss(j+1, 1) = X_miss(j+1, 1) * I1(j);
        X_miss(j+1, 2) = X_miss(j+1, 2) * I2(j);
    end
end
X = X_miss;

NP = 40;
NP_1 = 40;
Lmin = 3;
LP = 5;

% discretization size
hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%number of iterations for each level
%Nl =  floor(20 * 2^(2*LP) * hl*KL + 1000);
Nl = [5000,5000,5000];
%store the acceptance rate
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
    Theta_trace{k, 1} = zeros(Nl(k),7);
    Theta_traceN{k,1} = zeros(Nl(k),7);
    ML_Theta_trace{k, 1} = zeros(Nl(k),7);
end

for i = 1:LP - Lmin
    Theta_trace_1{i,1} = zeros(Nl(i+1),7);
    Theta_trace_2{i,1} = zeros(Nl(i+1),7);

    Theta_trace_1N{i,1} = zeros(Nl(i+1),7);
    Theta_trace_2N{i,1} = zeros(Nl(i+1),7);

    H1_trace{i,1} = zeros(Nl(i+1),1);
    H2_trace{i,1} = zeros(Nl(i+1),1);
end


Theta_A = [-4.3,-7,-6.5,-4,-2,-2,1];
tic;

Theta_A_p = Theta_A;
Theta_SIG_p = [exp(Theta_A_p(1:6)),2/(1+exp(-Theta_A_p(7)))-1];
Z = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace{1,1}(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;
N_count_2 = 0;
N_count_last_2 = 0;

Sigma_A1 = 0.35*diag([4,3,3,4]);
Sigma_A2 = 1.5*diag([0.2,0.2,0.5]);

Sigma_Al1 = 0.3*diag([4,3,3,4]);
Sigma_Al2 = 1.5*diag([0.2,0.2,0.5]);

for iter = 1:Nl(1)
 
    if mod(iter, 50) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)), ', ', num2str(Theta_A_p(3)), ', ', num2str(Theta_A_p(4)), ', ', num2str(Theta_A_p(5)), ', ', num2str(Theta_A_p(6)),  ', ', num2str(Theta_A_p(7)), ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 50) == 0
        N_count_last_1 = N_count_1;
    end
    
    Theta_A_prime_1f = Theta_A_p(1:4);
    Theta_A_prime_1f = mvnrnd(Theta_A_prime_1f, Sigma_A1*Sigma_A1');
    Theta_A_prime_1 = [Theta_A_prime_1f,Theta_A_p(5:7)];
    Theta_SIG_prime_1 = [exp(Theta_A_prime_1(1:6)),2/(1+exp(-Theta_A_prime_1(7)))-1];
   
    Z_prime_1 = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_prime_1(1),Theta_SIG_prime_1(2),Theta_SIG_prime_1(3),Theta_SIG_prime_1(4),Theta_SIG_prime_1(5),Theta_SIG_prime_1(6),Theta_SIG_prime_1(7));
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
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace{1, 1}(iter,:) = Theta_A_p; 
        lZ = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 

    if mod(iter, 50) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_2/(iter))]);
        disp(['current new AC = ', num2str((N_count_2 - N_count_last_2)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)), ', ', num2str(Theta_A_p(3)), ', ', num2str(Theta_A_p(4)), ', ', num2str(Theta_A_p(5)), ', ', num2str(Theta_A_p(6)),  ', ', num2str(Theta_A_p(7)), ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 50) == 0
        N_count_last_2 = N_count_2;
    end
    
    Theta_A_prime2 = Theta_A_p(5:7);
    Theta_A_prime2 = mvnrnd(Theta_A_prime2, Sigma_A2*Sigma_A2');
    Theta_A_prime_2 = [Theta_A_p(1:4), Theta_A_prime2];
    Theta_SIG_prime_2 = [exp(Theta_A_prime_2(1:6)),2/(1+exp(-Theta_A_prime_2(7)))-1];

    Z_prime_2 = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_prime_2(1),Theta_SIG_prime_2(2),Theta_SIG_prime_2(3),Theta_SIG_prime_2(4),Theta_SIG_prime_2(5),Theta_SIG_prime_2(6),Theta_SIG_prime_2(7));
    lZ_prime_2 = Z_prime_2;
    l_pos_Theta_A_prime_2 = l_posterior(Theta_A_prime_2, lZ_prime_2);
 
    alpha_U2 = min(0, l_pos_Theta_A_prime_2 - l_pos_Theta_A_p);
    U2 = log(rand);
    
    if U2 < alpha_U2
        Theta_A_p = Theta_A_prime_2;
        Theta_SIG_p = Theta_SIG_prime_2;
        lZ = lZ_prime_2;
        l_pos_Theta_A_p = l_pos_Theta_A_prime_2;
        Theta_trace{1, 1}(iter,:) = Theta_A_prime_2; 
        N_count_2 = N_count_2 + 1;        
    else
        Theta_trace{1, 1}(iter,:) = Theta_A_p; 
        lZ = PF_DB(Lmin, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 


end

Aln(1,1) = N_count_1 / Nl(1);
Aln(1,2) = N_count_2 / Nl(1);
toc;

H1_sum = 0;
H2_sum = 0;
tic;

%mlpmmh
for l = 1:LP - Lmin 

    level = l + Lmin;
    fprintf('level = %f\n', level);

   %Theta_l = mean(Theta_trace{1,1});
    Theta_l = [-4,-7,-7,-7,-2,-2,1];
    Theta_SIG_l = [exp(Theta_l(1:6)),2/(1+exp(-Theta_l(7)))-1];

    [H1_l, H2_l, G_l] = c_pf_db(level, NP_1, T, T_obs, X, Theta_SIG_l(1),Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4),Theta_SIG_l(5),Theta_SIG_l(6),Theta_SIG_l(7));                      
    lG_l = G_l;
    l_pos_theta_l = l_posterior(Theta_l, lG_l);
    
    N_count_l1 = 0;
    N_count_l2 = 0;

    for iter = 1:Nl(l+1)

        if mod(iter, 50) == 0
            fprintf('iter = %f\n', iter);
            fprintf('AR = %f\n', N_count_l1/iter);
            fprintf('H1 average = %f\n', H1_sum/50);
            fprintf('H2 average = %f\n', H2_sum/50);
            disp(['current estimate = [', num2str(Theta_l(1)), ', ', num2str(Theta_l(2)), ', ', num2str(Theta_l(3)), ', ', num2str(Theta_l(4)), ', ', num2str(Theta_l(5)), ', ', num2str(Theta_l(6)),  ', ', num2str(Theta_l(7)), ']']);
            H1_sum = 0;
            H2_sum = 0;
            toc;
            tic;
        end
        
            

        Theta_l_prime1 = Theta_l(1:4);
        Theta_l_prime1 = mvnrnd(Theta_l_prime1,Sigma_Al1*Sigma_Al1');
        Theta_l_prime_1 = [Theta_l_prime1, Theta_l(5:7)];
        Theta_l_SIG_prime_1 = [exp(Theta_l_prime_1(1:6)),2/(1+exp(-Theta_l_prime_1(7)))-1];

        [H1_lp1, H2_lp1, lG_lp1] = c_pf_db(level, NP_1, T, T_obs, X, Theta_l_SIG_prime_1(1),Theta_l_SIG_prime_1(2),Theta_l_SIG_prime_1(3),Theta_l_SIG_prime_1(4),Theta_l_SIG_prime_1(5),Theta_l_SIG_prime_1(6),Theta_l_SIG_prime_1(7));
        l_pos_theta_l_prime1 = l_posterior(Theta_l_prime_1, lG_lp1);
        alpha_l1 = min(0, l_pos_theta_l_prime1 - l_pos_theta_l);

        Ul1 = log(rand);
        if Ul1 < alpha_l1
            Theta_l = Theta_l_prime_1;
            Theta_SIG_l = Theta_l_SIG_prime_1;
            Theta_trace{l+1, 1}(iter,:) = Theta_l_prime_1;
            lG_l = lG_lp1;
            l_pos_theta_l = l_pos_theta_l_prime1;
            H1_l = H1_lp1;
            H2_l = H2_lp1;
            H1_trace{l, 1}(iter,1) = H1_lp1;
            H2_trace{l, 1}(iter,1) = H2_lp1;
            N_count_l1= N_count_l1 + 1;
            H1_sum = H1_sum + H1_lp1;
            H2_sum = H2_sum + H2_lp1;
        else
            Theta_trace{l+1, 1}(iter,:) = Theta_l; 
            [H1_l, H2_l, lG_l] = c_pf_db(level, NP_1, T, T_obs, X, Theta_SIG_l(1),Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4), Theta_SIG_l(5),Theta_SIG_l(6),Theta_SIG_l(7));
            l_pos_theta_l = l_posterior(Theta_l, lG_l);
            H1_trace{l, 1}(iter,1) = H1_l;
            H2_trace{l, 1}(iter,1) = H2_l;
            H1_sum = H1_sum + H1_l;
            H2_sum = H2_sum + H2_l;
        end   


        if mod(iter, 50) == 0
            fprintf('iter = %f\n', iter);
            fprintf('AR = %f\n', N_count_l2/iter);
            fprintf('H1 average = %f\n', H1_sum/50);
            fprintf('H2 average = %f\n', H2_sum/50);
            disp(['current estimate = [', num2str(Theta_l(1)), ', ', num2str(Theta_l(2)), ', ', num2str(Theta_l(3)), ', ', num2str(Theta_l(4)), ', ', num2str(Theta_l(5)), ', ', num2str(Theta_l(6)),  ', ', num2str(Theta_l(7)), ']']);
            H1_sum = 0;
            H2_sum = 0;
            toc;
            tic;
        end
        
    
        Theta_l_prime2 = Theta_l(5:7);
        Theta_l_prime2 = mvnrnd(Theta_l_prime2,Sigma_Al2*Sigma_Al2');
        Theta_l_prime_2 = [Theta_l(1:4), Theta_l_prime2];
        Theta_l_SIG_prime_2 = [exp(Theta_l_prime_2(1:6)),2/(1+exp(-Theta_l_prime_2(7)))-1];

        [H1_lp2, H2_lp2, lG_lp2] = c_pf_db(level, NP_1, T, T_obs, X, Theta_l_SIG_prime_2(1),Theta_l_SIG_prime_2(2),Theta_l_SIG_prime_2(3),Theta_l_SIG_prime_2(4),Theta_l_SIG_prime_2(5),Theta_l_SIG_prime_2(6),Theta_l_SIG_prime_2(7));
        l_pos_theta_l_prime2 = l_posterior(Theta_l_prime_2, lG_lp2);
        alpha_l2 = min(0, l_pos_theta_l_prime2 - l_pos_theta_l);

        Ul2 = log(rand);
        if Ul2 < alpha_l2

            Theta_l = Theta_l_prime_2;
            Theta_SIG_l = Theta_l_SIG_prime_2;
            Theta_trace{l+1, 1}(iter,:) = Theta_l_prime_2;
            lG_l = lG_lp2;
            l_pos_theta_l = l_pos_theta_l_prime2;
            H1_l = H1_lp2;
            H2_l = H2_lp2;
            H1_trace{l, 1}(iter,1) = H1_lp2;
            H2_trace{l, 1}(iter,1) = H2_lp2;
            N_count_l2= N_count_l2 + 1;
            H1_sum = H1_sum + H1_lp2;
            H2_sum = H2_sum + H2_lp2;
        else
            Theta_trace{l+1, 1}(iter,:) = Theta_l; 
            [H1_l, H2_l, lG_l] = c_pf_db(level, NP_1, T, T_obs, X, Theta_SIG_l(1),Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4), Theta_SIG_l(5),Theta_SIG_l(6),Theta_SIG_l(7));
            l_pos_theta_l = l_posterior(Theta_l, lG_l);
            H1_trace{l, 1}(iter,1) = H1_l;
            H2_trace{l, 1}(iter,1) = H2_l;
            H1_sum = H1_sum + H1_l;
            H2_sum = H2_sum + H2_l;
        end   

    end

        Aln(l+1,1) = N_count_l1/ Nl(l+1);
        Aln(l+1,2) = N_count_l2 / Nl(l+1);
end

toc;

burnin = 1;
for ll = 1:LP - Lmin

    for i = 1:7      
        Theta_trace_1N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))));
        Theta_trace_2N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H2_trace{ll,1}(:,1)) / sum(exp(H2_trace{ll,1}(:,1))));
        Theta_traceN{ll+1,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))) - exp(H2_trace{ll,1}(:,1))/ sum(exp(H2_trace{ll,1}(:,1))));
        ML_Theta_trace{ll+1,1}(:,i) = cumsum(Theta_traceN{ll+1, 1}(:,i)) ./ (1:Nl(ll+1))';
    end
end

final_theta =  mean(Theta_trace{1,1}(burnin:end,:));
level_means = zeros(LP-Lmin, 7);

for i=1:7
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



function [lwf, lwc, cz] = c_pf_db(L, NP, T, T_obs, X,a1,a2,a3,a4,sig1,sig2,rho)

    hlf = 2^(-L);
    hlc = 2^(-(L-1));

    Xf_est = zeros(NP, 2, T);
    Xf_est(:,1,1) = X(1,1) * ones(NP,1);
    Xf_est(:,2,1) = X(1,2) * ones(NP,1);

    Xc_est = zeros(NP, 2, T);
    Xc_est(:,1,1) = X(1,1) * ones(NP,1);
    Xc_est(:,2,1) = X(1,2) * ones(NP,1);

  
    lGLf1 = zeros(NP,T-1);
    lGLf2 = zeros(NP,T-1);
    lGLf3 = zeros(NP,T-1);
    lGLf = zeros(NP,T-1);

    lGLc1 = zeros(NP,T-1);
    lGLc2 = zeros(NP,T-1);
    lGLc3 = zeros(NP,T-1);
    lGLc = zeros(NP,T-1);

    lGLJ = zeros(NP,T-1);
    lGL_star = zeros(1,T-1);

    lwf = 0;
    lwc = 0;
  
     for k = 1:T-2

        XKf = zeros(NP,2);
        XKMf = zeros(NP,2);
        XKc = zeros(NP,2);
        XKMc = zeros(NP,2);
        
        XKMf(:,1) = Xf_est(:,1,k);
        XKMf(:,2) = Xf_est(:,2,k);
        XKMc(:,1) = Xc_est(:,1,k);
        XKMc(:,2) = Xc_est(:,2,k);
        steps = T_obs(k+1) - T_obs(k);
        
        nlf = ceil(steps/hlf);
        nlc = ceil(steps/hlc);
       
        
        if X(k+1,1) == 0 && X(k+1,2) ~= 0

            Xf_est(:,2,k+1) = X(k+1,2) * ones(NP,1);
            [samp_f, logf_den] = h_log_normal(NP, steps,a1,a2,a3,a4,sig1,sig2, rho, XKMf(:,1), XKMf(:,2), -1, X(k+1,2));
            for n = 1:NP 
                Xf_est(n,1,k+1) = samp_f(1,n);
                lGLf2(n,k) = logf_den(n);
            end
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,2,k+1) = X(k+1,2) * ones(NP,1);
             [samp_c, logc_den] = h_log_normal(NP, steps,a1,a2,a3,a4,sig1,sig2, rho, XKMc(:,1), XKMc(:,2), -1, X(k+1,2)); 
            for n = 1:NP 
                Xc_est(n,1,k+1) = samp_c(1,n);
                lGLc2(n,k) = logc_den(n);
            end
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);

            [logf_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,sig1,sig2,rho, XKMf(:,1), XKMf(:,2), XKf(:,1), XKf(:,2),XKMf(:,1), XKMf(:,2));
            for n = 1:NP 
                lGLf1(n,k) = logf_density(n);
            end

            [logc_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,sig1,sig2,rho, XKMc(:,1), XKMc(:,2), XKc(:,1), XKc(:,2),XKMc(:,1), XKMc(:,2));
            for n = 1:NP 
                lGLc1(n,k) = logc_density(n);
            end

            
            [Xf_Bri, Xc_Bri, derif_logx, deric_logx] = cou_em_linear(XKMf, XKf, XKMc,XKc,T_obs(k+1)-T_obs(k),  L, NP, a1,a2,a3,a4,sig1,sig2,rho);         
           
            Rfc = Xf_Bri(:, 1,1);
            Ffc = Xf_Bri(:, 2,1);
            Rfb = Xf_Bri(:, 1,1);
            Ffb = Xf_Bri(:, 2,1);
            Rfe = XKf(:, 1);
            Ffe = XKf(:, 2);
            % the integral of L
            for m = 0:nlf-2 
                dif_muf = zeros(NP,2);
                tfc = m*hlf;
                tf1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_muf(n,:) = [a2*Rfc(n)*(Ffb(n)- Ffc(n) + tfc/tf1*(Ffe(n)-Ffb(n))), a3*Ffc(n)*(Rfc(n)- Rfb(n)+ tfc/tf1*(Rfb(n)-Rfe(n)))];
                    drif_f = derif_logx(n,:,m+1);
                    lGLf3(n,k) = lGLf3(n,k) + hlf*(dif_muf(n,:)*drif_f');
                end
                if m ~= nlf-2
                    Rfc = Xf_Bri(:,1,m+2);
                    Ffc = Xf_Bri(:,2,m+2);
                end
            end
           
            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

            Rcc = Xc_Bri(:,1,1);
            Fcc = Xc_Bri(:,2,1);
            Rcb = Xc_Bri(:,1,1);
            Fcb = Xc_Bri(:,2,1);
            Rce = XKc(:,1);
            Fce = XKc(:,2);
            % the integral of L
            for m2 = 0:nlc-2 
                dif_muc = zeros(NP,2);
                tcc = m2*hlc;
                tc1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_muc(n,:) = [a2*Rcc(n)*(Fcb(n)- Fcc(n) + tcc/tc1*(Fce(n)-Fcb(n))), a3*Fcc(n)*(Rcc(n)- Rcb(n)+ tcc/tc1*(Rcb(n)-Rce(n)))];
                    drif_c= deric_logx(n,:,m2+1);
                    lGLc3(n,k) = lGLc3(n,k) + hlc*(dif_muc(n,:)*drif_c');
                end

                if m2 ~= nlc-2
                    Rcc = Xf_Bri(:,1,m2+2);
                    Fcc = Xf_Bri(:,2,m2+2);
                end
            end
            
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
        
           elseif X(k+1,2) == 0 && X(k+1,1) ~= 0

                    Xf_est(:,1,k+1) = X(k+1,1) * ones(NP,1);
                     [samp_f, logf_den] = h_log_normal(NP, steps,a1,a2,a3,a4,sig1,sig2, rho, XKMf(:,1), XKMf(:,2),  X(k+1,1), -1);
                    for n = 1:NP 
                        Xf_est(n,2,k+1) = samp_f(2,n);
                        lGLf2(n,k) = logf_den(n);
                    end
                    XKf(:,1) = Xf_est(:,1,k+1);
                    XKf(:,2) = Xf_est(:,2,k+1);
        
                    Xc_est(:,1,k+1) = X(k+1,1) * ones(NP,1);
                    [samp_c, logc_den] = h_log_normal(NP, steps,a1,a2,a3,a4,sig1,sig2, rho, XKMc(:,1), XKMc(:,2),  X(k+1,1), -1);
                    for n = 1:NP 
                        Xc_est(n,2,k+1) = samp_c(2,n);
                        lGLc2(n,k) = logc_den(n);
                    end
                    XKc(:,1) = Xc_est(:,1,k+1);
                    XKc(:,2) = Xc_est(:,2,k+1);
        
        
                    [logf_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,sig1,sig2,rho, XKMf(:,1), XKMf(:,2), XKf(:,1), XKf(:,2),XKMf(:,1), XKMf(:,2));
                    for n = 1:NP 
                        lGLf1(n,k) = logf_density(n);
                    end
        
                    [logc_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,sig1,sig2,rho, XKMc(:,1), XKMc(:,2), XKc(:,1), XKc(:,2),XKMc(:,1), XKMc(:,2));
                    for n = 1:NP 
                        lGLc1(n,k) = logc_density(n);
                    end
            
                    [Xf_Bri, Xc_Bri, derif_logx, deric_logx] = cou_em_linear(XKMf, XKf, XKMc,XKc,T_obs(k+1)-T_obs(k),  L, NP, a1,a2,a3,a4,sig1,sig2,rho);         
                   
                    Rfc = Xf_Bri(:, 1,1);
                    Ffc = Xf_Bri(:, 2,1);
                    Rfb = Xf_Bri(:, 1,1);
                    Ffb = Xf_Bri(:, 2,1);
                    Rfe = XKf(:, 1);
                    Ffe = XKf(:, 2);
                    % the integral of L
                    for m = 0:nlf-2 
                        dif_muf = zeros(NP,2);
                        tfc = m*hlf;
                        tf1 = T_obs(k+1) - T_obs(k);
                        for n = 1:NP
                            dif_muf(n,:) = [a2*Rfc(n)*(Ffb(n)- Ffc(n) + tfc/tf1*(Ffe(n)-Ffb(n))), a3*Ffc(n)*(Rfc(n)- Rfb(n)+ tfc/tf1*(Rfb(n)-Rfe(n)))];
                            drif_f = derif_logx(n,:,m+1);
                            lGLf3(n,k) = lGLf3(n,k) + hlf*(dif_muf(n,:)*drif_f');
                        end
                        if m ~= nlf-2
                            Rfc = Xf_Bri(:,1,m+2);
                            Ffc = Xf_Bri(:,2,m+2);
                        end
                    end
           
                    lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

                    Rcc = Xc_Bri(:,1,1);
                    Fcc = Xc_Bri(:,2,1);
                    Rcb = Xc_Bri(:,1,1);
                    Fcb = Xc_Bri(:,2,1);
                    Rce = XKc(:,1);
                    Fce = XKc(:,2);
                    % the integral of L
                    for m2 = 0:nlc-2 
                        dif_muc = zeros(NP,2);
                        tcc = m2*hlc;
                        tc1 = T_obs(k+1) - T_obs(k);
                        for n = 1:NP
                            dif_muc(n,:) = [a2*Rcc(n)*(Fcb(n)- Fcc(n) + tcc/tc1*(Fce(n)-Fcb(n))), a3*Fcc(n)*(Rcc(n)- Rcb(n)+ tcc/tc1*(Rcb(n)-Rce(n)))];
                            drif_c= deric_logx(n,:,m2+1);
                            lGLc3(n,k) = lGLc3(n,k) + hlc*(dif_muc(n,:)*drif_c');
                        end
        
                        if m2 ~= nlc-2
                            Rcc = Xf_Bri(:,1,m2+2);
                            Fcc = Xf_Bri(:,2,m2+2);
                        end
                    end
                    
                    lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
        
                    %lGLJ(:,k) = max(lGLf(:,k),lGLc(:,k));
                    lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
                      
        elseif X(k+1,2) ~= 0 && X(k+1,1) ~= 0
            Xf_est(:,1,k+1) = X(k+1,1)*ones(NP,1);
            Xf_est(:,2,k+1) = X(k+1,2)*ones(NP,1);
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,1,k+1) = X(k+1,1)*ones(NP,1);
            Xc_est(:,2,k+1) = X(k+1,2)*ones(NP,1);
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);

             [logf_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,sig1,sig2,rho, XKMf(:,1), XKMf(:,2), XKf(:,1), XKf(:,2),XKMf(:,1), XKMf(:,2));
            for n = 1:NP 
                lGLf1(n,k) = logf_density(n);
            end

            lGLf2(:,k) = 0;

            [Xf_Bri, Xc_Bri, derif_logx, deric_logx] = cou_em_linear(XKMf, XKf, XKMc,XKc,T_obs(k+1)-T_obs(k),  L, NP, a1,a2,a3,a4,sig1,sig2,rho);         
            Rfc = Xf_Bri(:, 1,1);
            Ffc = Xf_Bri(:, 2,1);
            Rfb = Xf_Bri(:, 1,1);
            Ffb = Xf_Bri(:, 2,1);
            Rfe = XKf(:, 1);
            Ffe = XKf(:, 2);
            % the integral of L
            for m = 0:nlf-2 
                dif_muf = zeros(NP,2);
                tfc = m*hlf;
                tf1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_muf(n,:) = [a2*Rfc(n)*(Ffb(n)- Ffc(n) + tfc/tf1*(Ffe(n)-Ffb(n))), a3*Ffc(n)*(Rfc(n)- Rfb(n)+ tfc/tf1*(Rfb(n)-Rfe(n)))];
                    drif_f = derif_logx(n,:,m+1);
                    lGLf3(n,k) = lGLf3(n,k) + hlf*(dif_muf(n,:)*drif_f');
                end
                if m ~= nlf-2
                    Rfc = Xf_Bri(:,1,m+2);
                    Ffc = Xf_Bri(:,2,m+2);
                end
            end
           
            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

             [logc_density,~] = linear_tran_t_aux(NP, 0, T_obs(k+1)-T_obs(k), a1,a2,a3,a4,sig1,sig2,rho, XKMc(:,1), XKMc(:,2), XKc(:,1), XKc(:,2),XKMc(:,1), XKMc(:,2));
            for n = 1:NP 
                lGLc1(n,k) = logc_density(n);
            end

            lGLc2(:,k) = 0;

            Rcc = Xc_Bri(:,1,1);
            Fcc = Xc_Bri(:,2,1);
            Rcb = Xc_Bri(:,1,1);
            Fcb = Xc_Bri(:,2,1);
            Rce = XKc(:,1);
            Fce = XKc(:,2);
            % the integral of L
            for m2 = 0:nlc-2 
                dif_muc = zeros(NP,2);
                tcc = m2*hlc;
                tc1 = T_obs(k+1) - T_obs(k);
                for n = 1:NP
                    dif_muc(n,:) = [a2*Rcc(n)*(Fcb(n)- Fcc(n) + tcc/tc1*(Fce(n)-Fcb(n))), a3*Fcc(n)*(Rcc(n)- Rcb(n)+ tcc/tc1*(Rcb(n)-Rce(n)))];
                    drif_c= deric_logx(n,:,m2+1);
                    lGLc3(n,k) = lGLc3(n,k) + hlc*(dif_muc(n,:)*drif_c');
                end

                if m2 ~= nlc-2
                    Rcc = Xf_Bri(:,1,m2+2);
                    Fcc = Xf_Bri(:,2,m2+2);
                end
            end
            
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);

            %lGLJ(:,k) = max(lGLf(:,k),lGLc(:,k));
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));

        end

        GL0 = exp(lGLJ(:,k) - max(lGLJ(:,k)));
        lGL_star(1,k)= log(sum(GL0)) + max(lGLJ(:,k));
        lwf = lwf + mean((lGLf(:,k) - lGLJ(:,k)));
        lwc = lwc + mean((lGLc(:,k) - lGLJ(:,k)));
        GLL = GL0 / sum(GL0);
       
            if isnan(sum(GLL)) 
                disp('ANNOYING NAN ERROR! GLL');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end
            if  sum(GLL) == 0
                disp(' GLL = 0');
                cz = -inf;
                lwf = 0;
                lwc = 0;
                return
            end
            
    
        I = resampleSystematic( GLL);
        Xf_est(:,1,1:k) = Xf_est(I, 1,1:k);
        Xf_est(:,2,1:k) = Xf_est(I, 2,1:k);
        %lGLf(:,1:k) = lGLf(I,1:k);
        Xc_est(:,1,1:k) = Xc_est(I, 1,1:k);
        Xc_est(:,2,1:k) = Xc_est(I, 2,1:k);   
     end
     
    cz = (T-1) * log(1/NP) + sum(lGL_star);     
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


function [Xf_bridge,Xc_bridge, derif_x, deric_x] = cou_em_linear(Xf_start, Xf_end, Xc_start,Xc_end,dis, L, NP, a11,a12,a21,a22,sig1,sig2,rho)
  
    hhf = 2^(-L);
    nhf = ceil(dis/(hhf));
    hhc = 2^(-(L-1));
    nhc = ceil(dis/hhc);
    Xf_bridge = zeros(NP,2,nhf);
    Xf_bridge(:,:,1) = Xf_start;
    Xc_bridge = zeros(NP,2,nhc);
    Xc_bridge(:,:,1) = Xc_start;
    
    Rfe = Xf_end(:,1);
    Ffe = Xf_end(:,2);
    Rfb = Xf_start(:,1);
    Ffb = Xf_start(:,2);

    Rce = Xc_end(:,1);
    Fce = Xc_end(:,2);
    Rcb = Xc_start(:,1);
    Fcb = Xc_start(:,2);

    derif_x = zeros(NP,2,nhf);
    deric_x = zeros(NP,2,nhc);

    lg_Xf_bridge = zeros(NP,2,nhf);
    lg_Xf_bridge(:,:,1) = log(Xf_start);
    
    lg_Xc_bridge = zeros(NP,2,nhc);
    lg_Xc_bridge(:,:,1) = log(Xc_start);
    
    dWf = zeros(NP,2,nhf-1);
    dWc = zeros(NP,2,nhc-1);
    

    for ii = 1:nhc-1 
        dWc(:,:,ii) = zeros(NP, 2);
        for jj = 1:2

                %generate the correlated brownian motion
                Zf1 = randn(NP, 1);
                Zf2 = randn(NP, 1);
                Zf2_corr = rho * Zf1 + sqrt(1 - rho^2) * Zf2;
                Zf1_corr = Zf1;
                dWf1 = sqrt(hhf) * Zf1_corr;
                dWf2 = sqrt(hhf) * Zf2_corr;
                dWf (:,:,2*(ii-1)+jj)= [dWf1,dWf2];
               
                % Take log(Rt) log(Ft) and then take exp
        
                Rfs = Xf_bridge(:,1,2*(ii-1) + jj);
                Ffs = Xf_bridge(:,2,2*(ii-1) + jj);    
        
                [~,grad_f_densities] = linear_tran_t_aux(NP, (2*(ii-1)+jj-1)*hhf,dis, a11,a12, a21,a22, sig1, sig2,rho, Rfb, Ffb, Rfe, Ffe, Rfs, Ffs);
                derif_x(:,:,2*(ii-1) + jj) = grad_f_densities';

                for n = 1:NP
                    drif_t1 = [a11 - a12*Ffs(n); a21*Rfs(n) - a22]*hhf;
                    drif_t2 = [sig1^2*Rfs(n),rho*sig1*sig2*Ffs(n);rho*sig1*sig2*Rfs(n),sig2^2*Ffs(n)]*derif_x(n,:,2*(ii-1)+jj)'*hhf;
                    drif_t3 = [-sig1^2/2;-sig2^2/2]*hhf;
                    diff_t = [sig1,0;0,sig2]*dWf(n,:, 2*(ii-1)+jj)';
                    lg_Xf_bridge(n,:,2*(ii-1)+jj+1) = lg_Xf_bridge(n,:,2*(ii-1) + jj) + (drif_t1 + drif_t2 + drif_t3 + diff_t)';   
                    Xf_bridge(n,:,2*(ii-1)+jj+1) = exp(lg_Xf_bridge(n,:,2*(ii-1)+jj+1));
                end
            dWc(:,:,ii) = dWc(:,:,ii) + dWf(:,:,2*(ii-1)+jj);
        end

        Rcs = Xc_bridge(:,1,ii);
        Fcs = Xc_bridge(:,2,ii);    

        [~,grad_c_densities] = linear_tran_t_aux(NP, (ii-1)*hhc,dis, a11,a12, a21,a22, sig1, sig2,rho, Rcb, Fcb, Rce, Fce, Rcs, Fcs);
        deric_x(:,:,ii) = grad_c_densities';

        for n = 1:NP
            dric_t1 = [a11 - a12*Fcs(n); a21*Rcs(n) - a22]*hhc;
            dric_t2 = [sig1^2*Rcs(n),rho*sig1*sig2*Fcs(n);rho*sig1*sig2*Rcs(n),sig2^2*Fcs(n)]*deric_x(n,:,ii)'*hhc;
            dric_t3 = [-sig1^2/2;-sig2^2/2]*hhc;
            difc_t = [sig1,0;0,sig2]*dWc(n,:,ii)';
            lg_Xc_bridge(n,:,ii+1) = lg_Xc_bridge(n,:,ii) + (dric_t1 + dric_t2 + dric_t3 + difc_t)';   
            Xc_bridge(n,:,ii+1) = exp(lg_Xc_bridge(n,:,ii+1));
        end

    end
end


function [samples, log_density] = h_log_normal(N, tdiff, a11,a12, a21,a22, sig1, sig2, rho, R0, F0, Re, Fe)
    samples = zeros(2,N);
    log_density = zeros(1,N);
    if (Re == -1)&&(Fe ~= -1)
        for n = 1:N
            sigma = 0.05;
            mu = log(R0(n)) - sigma^2/2;
            X = lognrnd(mu, sigma);
            samples(1,n) = X;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
            samples(2,n) = Fe;
        end
    end

    if (Fe == -1)&&(Re~=-1)
        for n = 1:N
            sigma = 0.05;
            mu = log(F0(n)) - sigma^2/2;
            X = lognrnd(mu, sigma);
            samples(2,n) = X;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
            samples(1,n) = Re;
        end
    end
    if (Fe == -1)&&(Re==-1)
        for n = 1:N
            sigma = 0.05;
            mur = log(R0(n)) - sigma^2/2;
            Xr = lognrnd(mur, sigma);
            samples(1,n) = Xr;
            muf = log(F0(n)) - sigma^2/2;
            Xf = lognrnd(muf, sigma);
            samples(2,n) = Xf;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
           
        end
    end
end

function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end


function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior =  lG(Theta(1),-4,1)+lG(Theta(2),-6,1)+lG(Theta(3),-7,1)+lG(Theta(4),-7,1)+lG(Theta(5),-2,1)+lG(Theta(6),-2,1)+lG(Theta(7),1,1);
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
