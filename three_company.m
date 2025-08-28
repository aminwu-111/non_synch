% This code is used to estimate the drift term of the SLV model by the
% using the PMCMC method
%the model is 
% dR = R * (alpha - beta*F) * dt + sigma_1*R * dW1(t)
% dF = F * (delta*R - gamma) * dt + sigma_2*F * dW2(t)
% dW1 and dW2 are N(0,dt) with correlation rho
%%MLPMCMC with diffusion bridge, estimating drift coefficient. 
%%observation new irregularly observed, not in unit time, exact trajectory
warning('off', 'all');
%{
close all;
clear;
clc;
%format long
warning('off', 'all');


% Parameter settings
maxT = 30;            % Total time
tarFreq = 2; 
MI = 1 / tarFreq; 
numP = 300;
TI = exprnd(MI, numP, 1);
cumT = cumsum(TI);
validTPs = cumT(cumT <= maxT);
if validTPs(end) ~= maxT
    T_Fsimu = [validTPs; maxT];
end
T_num = length(T_Fsimu);
Tdiffs = [TI(1:T_num-1); maxT - validTPs(T_num-1)];


L = 12;
dt = 2^(-L);          % Time step
nh = ceil(Tdiffs/dt);

X = zeros(T_num+1, 2);
%X(1,:) = [140, 14];
X(1,:) = [1.40, 0.14];

% Model parameters
alpha = 1;         % Prey birth rate 
beta = 2;          % Predation rate
delta = 1.5;        % Predator growth rate
gamma = 1;         % Predator death rate
sigma_1 = 0.3;
sigma_2 = 0.3;
rho = 0.5;  
%rho = 0.0;  

for k = 1: T_num
    Xde = zeros(1,2);
    Xde(1,1) = X(k, 1);
    Xde(1,2) = X(k, 2);
      
    Z1 = randn(nh(k), 1);
    Z2 = randn(nh(k), 1);
    Z2_corr = rho * Z1 + sqrt(1 - rho^2) * Z2;
    Z1_corr = Z1;

    for i = 1:nh(k)-1  
        R = Xde(1);  % Current prey population
        F = Xde(2);  % Current predator population
   
         % Deterministic part
        dXt = [ alpha * R - beta * R * F;             % dR (change in prey)
                delta * R * F - gamma * F ];          % dF (change in predator)
        % Stochastic part
        dW1 = sqrt(dt) * Z1_corr(i);
        dW2 = sqrt(dt) * Z2_corr(i); % Using generated correlated Brownian increments        
        dW = [dW1; dW2];
        G = [
            sigma_1*R, 0;    % dW1 related term for prey
            0, sigma_2*F     % dW2 related term for predator
        ];
    
        % Update state
        Xde = Xde + dXt' * dt + (G * dW)' ; 
    end

    dis_E = Tdiffs(k)-(nh(k)-1)*dt;
    RE = Xde(1);  % Current prey population
    FE = Xde(2);  % Current predator population

     % Deterministic part
    dXtE = [ alpha * RE - beta * RE * FE;             % dR (change in prey)
            delta * RE * FE - gamma * FE ];          % dF (change in predator)
    
    % Stochastic part
    dWE = [sqrt(dis_E) * Z1(end); sqrt(dis_E) * Z2(end)];  % Using generated correlated Brownian increments
    
    GE = [
        sigma_1*RE, 0;    % dW1 related term for prey
        0, sigma_2*FE     % dW2 related term for predator
    ];

    % Update state
    Xde= Xde + dXtE' * dis_E + (GE * dWE)' ; 
    X(k+1,:) = Xde;
end


% Plotting results
figure;
subplot(2, 1, 1);
plot([0;T_Fsimu], X(:, 1), 'b', 'LineWidth', 1.5);
title('Prey Population over Time (With Correlated Brownian Motions)');
xlabel('Time');
ylabel('Prey Population');

subplot(2, 1, 2);
plot([0;T_Fsimu], X(:, 2), 'r', 'LineWidth', 1.5);
title('Predator Population over Time (With Correlated Brownian Motions)');
xlabel('Time');
ylabel('Predator Population');

sgtitle('Stochastic Lotka-Volterra Model');

figure;
plot(X(:, 1), X(:, 2), '-');

X_miss = X;

I1 = randi([0,1],1,T_num-1);
I2 = randi([0,1],1,T_num-1);
for j = 1:T_num-1
    if I1(j) == 0 && I2(j) == 0
        index = randi([1,2]);
        X_miss(j+1, index) = 0;
    else
        X_miss(j+1, 1) = X_miss(j+1, 1) * I1(j);
        X_miss(j+1, 2) = X_miss(j+1, 2) * I2(j);
    end
end
X_miss(end, :) = 0;

T_obs = [0;T_Fsimu];


figure;
subplot(2, 1, 1);
plot([0;T_Fsimu], X_miss(:, 1), 'b*', 'LineWidth', 1.5);
title('Prey Population over Time (With Correlated Brownian Motions)');
xlabel('Time');
ylabel('Prey Population');

subplot(2, 1, 2);
plot([0;T_Fsimu], X_miss(:, 2), 'r*', 'LineWidth', 1.5);
title('Predator Population over Time (With Correlated Brownian Motions)');
xlabel('Time');
ylabel('Predator Population');

sgtitle('Stochastic Lotka-Volterra Model');


%error();

mkdir(['slvT_', num2str(maxT)]);
writematrix(T_obs,['slvT_', num2str(maxT), '/', num2str(tarFreq),'_times.txt']);

writematrix(X_miss,['slvT_', num2str(maxT), '/', num2str(tarFreq), '_data.txt']);

writematrix(X,['slvT_', num2str(maxT), '/', num2str(tarFreq),'_fulldata.txt']);

disp(['smalled level to use = ', num2str(-min(log2(T_obs(2:end) - T_obs(1:end-1))))]);
%error();

%}

clear
close all
clc
format long
%{
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

%}

X = load('ml_3_comp.mat').X_miss;
T = load('ml_3_comp.mat').T;
T_obs = load('ml_3_comp.mat').T_obs;
NP = 40;
%NP_1 = 50;
Lmin = 5;
LP = 5;

% discretization size
hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%number of iterations for each level
%Nl =  floor(20 * 2^(2*LP) * hl.^(9/10) + 1000);
Nl = 2000;
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

%Theta_A = [-0.04, 0.69, 0.43, -0.065, -0.93,-0.98,1.23];
Theta_A = [-5,-5,-6,-5,-2.5,-1.8,-1];
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

Sigma_A1 = 0.35*diag([4,4,3.5,4]);
Sigma_A2 = 1.5*diag([0.2,0.2,0.5]);

%Sigma_A1 = 0.1*diag([1.17,1.2,1.15,1.2]);
%Sigma_A2 = 0.25*diag([1,1,1.5]);

for iter = 1:Nl(1)
 
    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_1/(iter))]);
        disp(['current new AC = ', num2str((N_count_1 - N_count_last_1)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)), ', ', num2str(Theta_A_p(3)), ', ', num2str(Theta_A_p(4)), ', ', num2str(Theta_A_p(5)), ', ', num2str(Theta_A_p(6)),  ', ', num2str(Theta_A_p(7)), ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
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

    if mod(iter, 100) == 0
        fprintf('iter = %f\n', iter);
        disp(['current AC = ', num2str(N_count_2/(iter))]);
        disp(['current new AC = ', num2str((N_count_2 - N_count_last_2)/(mod(iter,50)+1))]);
        disp(['current estimate = [', num2str(Theta_A_p(1)), ', ', num2str(Theta_A_p(2)), ', ', num2str(Theta_A_p(3)), ', ', num2str(Theta_A_p(4)), ', ', num2str(Theta_A_p(5)), ', ', num2str(Theta_A_p(6)),  ', ', num2str(Theta_A_p(7)), ']']);
        disp(['estimated posterior = ', num2str(l_pos_Theta_A_p)]);
    end
    if mod(iter, 100) == 0
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

%{
toc;

H1_sum = 0;
H2_sum = 0;
tic;
%mlpmmh

for l = 1:LP - Lmin 

    level = l + Lmin;
    fprintf('level = %f\n', level);

    Theta_l = mean(Theta_trace{1,1});
    %Theta_l = [0.7,0.4,-0.6,1.7];

    [H1_l, H2_l, G_l] = c_pf_db(level, NP_1, T, T_obs, X_miss, Theta_l(1),Theta_l(2),Theta_l(3),Theta_l(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
                        
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
            
        Theta_l_prime1 = mvnrnd(Theta_l,Sigma_A1*Sigma_A1');
        Theta_l_prime = Theta_l_prime1;
        [H1_lp, H2_lp, lG_lp] = c_pf_db(level, NP_1, T, T_obs, X_miss, Theta_l_prime(1),Theta_l_prime(2),Theta_l_prime(3),Theta_l_prime(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
        l_pos_theta_l_prime = l_posterior(Theta_l_prime, lG_lp);
        alpha_l = min(0, l_pos_theta_l_prime - l_pos_theta_l);

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
            H1_sum = H1_sum + H1_lp;
            H2_sum = H2_sum + H2_lp;
        else
            Theta_trace{l+1, 1}(iter,:) = Theta_l; 
            [H1_l, H2_l, lG_l] = c_pf_db(level, NP_1, T, T_obs, X_miss, Theta_l(1),Theta_l(2),Theta_l(3),Theta_l(4), sqrt(COV(1,1)),sqrt(COV(2,2)),rho);
            l_pos_theta_l = l_posterior(Theta_l, lG_l);
            H1_trace{l, 1}(iter,1) = H1_l;
            H2_trace{l, 1}(iter,1) = H2_l;
            H1_sum = H1_sum + H1_l;
            H2_sum = H2_sum + H2_l;
        end   
    end


        Aln(l+1,1) = N_count_l / Nl(l+1);
end

toc;

burnin = 400;
for ll = 1:LP - Lmin

    for i = 1:4      
        Theta_trace_1N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))));
        Theta_trace_2N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H2_trace{ll,1}(:,1)) / sum(exp(H2_trace{ll,1}(:,1))));
        Theta_traceN{ll+1,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))) - exp(H2_trace{ll,1}(:,1))/ sum(exp(H2_trace{ll,1}(:,1))));
        ML_Theta_trace{ll+1,1}(:,i) = cumsum(Theta_traceN{ll+1, 1}(:,i)) ./ (1:Nl(ll+1))';
    end
end

Theta_traceN{1,1} = mean(Theta_trace{1,1}(burnin:end,:));

final_theta = Theta_traceN{1,1};
level_means = zeros(LP-Lmin, 4);

for i=1:4
    for j = 1:LP - Lmin
        final_theta(i) = final_theta(i) + sum(Theta_traceN{j+1,1}(burnin:end,i));
        level_means(j,i) = sum(Theta_traceN{j+1,1}(burnin:end,i));
    end
end
%}

burnin = 1;
figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,1), 'r--');
title('A11')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,2), 'r--');
title('A12')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,3), 'r--');
title('A21')
hold off
figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,4), 'r--');
title('A22')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,5), 'r--');
title('sig_1')
hold off
figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,6), 'r--');
title('sig_2')
hold off

figure
plot(burnin:Nl(1), Theta_trace{1,1}(burnin:end,7), 'r--');
title('rho')
hold off



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
       
        %{
        if X(k+1,2) ~= 0 && X(k+1,1) == 0

            X_est(:,2,k+1) = X(k+1,2) * ones(NP,1);
            [samples, log_densities] = OU_tran_h_gbm(NP, steps,a1,a2,a3,a4,0.1*s1,0.1*s2,rho,XKM(:,1), XKM(:,2), -1, X_est(:,2,k+1));
            X_est(:,1,k+1) = samples(1,:)';
            lGL2(:,k) = log_densities;
            XK(:,1) = X_est(:,1,k+1);
            XK(:,2) = X_est(:,2,k+1);
    
            [den_tilde,~] = GBM_tran_t_aux(NP, steps, a1,a2, a3,a4, s1, s2,rho, XKM(:,1), XKM(:,2), XK(:,1), XK(:,2),XKM(:,1), XKM(:,2));
            lGL1(:,k) = den_tilde';
            [X_Bri, deri_logx] = em_b_gbm(XKM, XK, steps, L, NP, a1,a2, a3,a4, s1, s2,rho);
            Rc = X_Bri(:,1,1);
            Fc = X_Bri(:,2,1);
            Rb = X_Bri(:,1,1);
            Fb = X_Bri(:,2,1);
            Re = XK(:,1);
            Fe = XK(:,2);

            % the integral of L
            for m = 0:nl-1
                Rc = X_Bri(:,1,m+1);
                Fc = X_Bri(:,2,m+1);
                %t_diff = steps - m*hl;
                dif_mu = zeros(NP,2);
                for n = 1:NP
                    dif_mu(n,:) = [a2*(-Rc(n)*Fc(n) + Rc(n)*Fb(n)),a3*(Rc(n)*Fc(n) - Rb(n)*Fc(n))];
                    dri_f = deri_logx(n,:,m+1);
                    lGL3(n,k) = lGL3(n,k) + hl*(dif_mu(n,:)*dri_f');
                    
                end
            end

            % Rce = X_Bri(:,1,nl);
            % Fce = X_Bri(:,2,nl);
            % Rbe = X_Bri(:,1,nl-1);
            % Fbe = X_Bri(:,2,nl-1);
            % t_diffe = steps - (nl-1)*hl;
            % dif_mue = zeros(NP,2);
            % %dri_fe = zeros(NP,2);
            % dif_ae = zeros(2,2,NP);
            % dri_ce = zeros(2,2,NP);
            % 
            % for n = 1:NP
            %     dif_mue(n,:) = [a2*(-Rce(n)*Fce(n) + Rce(n)*Fbe(n) + Rbe(n)*Fce(n) - Rbe(n)*Fbe(n)), a3*(Rce(n)*Fce(n) - Rce(n)*Fbe(n) - Rbe(n)*Fce(n) +Rbe(n)*Fbe(n))];
            %     muce = Mbri(n,:,nl);
            %     Cce = reshape(Cbri(n,:,nl),[2,2]);
            %     dri_fe = deri_logx(n,:,nl);
            %     dif_ae(:,:,n) = [s1^2*(Rce(n)^2-Re(n)^2),rho*s1*s2*(Rce(n)*Fce(n)-Re(n)*Fe(n));rho*s1*s2*(Rce(n)*Fce(n)-Re(n)*Fe(n)),s2^2*(Fce(n)^2-Fe(n)^2)];
            %     dri_ce(:,:,n) = expm(a_drift(:,:,n,nl)'*t_diffe)*inv(Cce)*expm(a_drift(:,:,n,nl)*t_diffe) - expm(a_drift(:,:,n,nl)'*t_diffe)*inv(Cce)*(XK(n,:) - muce)'*(XK(n,:) - muce)*inv(Cce)'*expm(a_drift(:,:,n,nl)*t_diffe);
            %     lGL3(n,k) = lGL3(n,k) + t_diffe*(dif_mue(n,:)*dri_fe' - 0.5*trace(dif_ae(:,:,n)*dri_ce(:,:,n)));
            %     if lGL3(n,k) > 1e+03
            %         %disp(m_s);
            %         %disp(c_s);
            %         %disp(['lGL3 > 1e+3, 2']);
            %     end
            % end
            
            lGL(:,k) = lGL3(:,k)+lGL2(:,k)+lGL1(:,k);
        %}
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
    log_prior =  lG(Theta(1),-5,3)+lG(Theta(2),-5,3)+lG(Theta(3),-6,3)+lG(Theta(4),-5,3)+lG(Theta(5),-2,1)+lG(Theta(6),-2,1)+lG(Theta(7),-1,1);
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
