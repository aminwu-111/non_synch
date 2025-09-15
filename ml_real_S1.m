% This code is used to estimate the drift term of the SLV model by the
% using the MLPMCMC method
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
X_unprocessed = readmatrix('beest_zebra.xlsx');
X_unprocessed(:,1) = X_unprocessed(:,1)-X_unprocessed(1,1);

T_obs = X_unprocessed(:,1);
X = [X_unprocessed(:,3),X_unprocessed(:,2)]/1000;
T_obs = T_obs(1:end-1);

%T_obs = readmatrix('slvT_25/2_times.txt');
%X = readmatrix('slvT_25/2_fulldata.txt');
%rho = 0.5;  
T= length(T_obs);

%{
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
%}

NP = 20;
NP_1 = 20;
Lmin = 3;
LP = 5;

% discretization size
hl = zeros(LP - Lmin + 1, 1);
hl(1) = 2^(-Lmin);
for l = 1:LP- Lmin
    hl(l+1) = 2^(-l- Lmin);
end

%number of iterations for each level
%Nl =  floor(20 * 2^(2*LP) * hl.^(9/10) + 1000);
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

%Theta_A = [-0.04, 0.69, 0.43, -0.065, -0.93,-0.98,1.23];
Theta_A = [-3.3,-5,-6.8,-4,-2,-2,1];
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

Sigma_Al1 = 0.5*diag([4,3,3,4]);
Sigma_Al2 = 2*diag([0.2,0.2,0.5]);

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

    Theta_l = mean(Theta_trace{1,1});
    %Theta_l = [ -3.693436164779128  -5.694320436305496  -6.998440850182257  -4.678891994294973  -2.166105264779996  -2.209833446267185   1.070644182284650];
    Theta_SIG_l = [exp(Theta_l(1:6)),2/(1+exp(-Theta_l(7)))-1];

    [H1_l, H2_l, G_l] = c_pf_db(level, NP_1, T, T_obs, X, Theta_SIG_l(1),Theta_SIG_l(2),Theta_SIG_l(3),Theta_SIG_l(4),Theta_SIG_l(5),Theta_SIG_l(6),Theta_SIG_l(7));                      
    lG_l = G_l;
    l_pos_theta_l = l_posterior(Theta_l, lG_l);
    
    N_count_l1 = 0;
    N_count_l2 = 0;
    for iter = 1:Nl(l+1)
        
        if mod(iter, 10) == 0
            
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
        
            

        Theta_l_prime_1 = Theta_l(1:4);
        Theta_l_prime_1 = mvnrnd(Theta_l_prime_1,Sigma_Al1*Sigma_Al1');
        Theta_l_prime_1 = [Theta_l_prime_1, Theta_l(5:7)];
        Theta_l_SIG_prime_1 = [exp(Theta_l_prime_1(1:6)),2/(1+exp(-Theta_l_prime_1(7)))-1];

        [H1_lp1, H2_lp1, lG_lp1] = c_pf_db(level, NP_1, T, T_obs, X, Theta_l_SIG_prime_1(1),Theta_l_SIG_prime_1(2),Theta_l_SIG_prime_1(3),Theta_l_SIG_prime_1(4),Theta_l_SIG_prime_1(5),Theta_l_SIG_prime_1(6),Theta_l_SIG_prime_1(7));
        l_pos_theta_l_prime1 = l_posterior(Theta_l_prime_1, lG_lp1);
        alpha_l1 = min(0, l_pos_theta_l_prime1 - l_pos_theta_l);


        Ul1 = log(rand);
        if Ul1 < alpha_l1

            Theta_l = Theta_l_prime_1;
            Theta_trace{l+1, 1}(iter,:) = Theta_l_prime_1;
            lG_l = lG_lp1;
            l_pos_theta_l = l_pos_theta_l_prime1;
            H1_l = H1_lp1;
            H2_l = H2_lp1;
            H1_trace{l, 1}(iter,1) = H1_lp1;
            H2_trace{l, 1}(iter,1) = H2_lp1;
            N_count_l1= N_count_l1 + 1;
            H1_sum = H1_sum + H1_lp1;
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


        if mod(iter, 10) == 0
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
        Theta_l_prime2 = mvnrnd(Theta_l_prime2,Sigma_A2*Sigma_A2');
        Theta_l_prime_2 = [Theta_l(1:4), Theta_l_prime2];
        Theta_l_SIG_prime_2 = [exp(Theta_l_prime_2(1:6)),2/(1+exp(-Theta_l_prime_2(7)))-1];

        [H1_lp2, H2_lp2, lG_lp2] = c_pf_db(level, NP_1, T, T_obs, X, Theta_l_SIG_prime_2(1),Theta_l_SIG_prime_2(2),Theta_l_SIG_prime_2(3),Theta_l_SIG_prime_2(4),Theta_l_SIG_prime_2(5),Theta_l_SIG_prime_2(6),Theta_l_SIG_prime_2(7));
        l_pos_theta_l_prime2 = l_posterior(Theta_l_prime_2, lG_lp2);
        alpha_l2 = min(0, l_pos_theta_l_prime2 - l_pos_theta_l);


        Ul2 = log(rand);
        if Ul2 < alpha_l2

            Theta_l = Theta_l_prime_2;
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

burnin = 100;
for ll = 1:LP - Lmin
    % SSS: Ad-Hoc removing outliers...
    m = median(H1_trace{ll,1}(:,1));
    H1_trace{ll,1}(:,1) = (abs(H1_trace{ll,1}(:,1) - m)/abs(m) < 3).*H1_trace{ll,1}(:,1);
    m = median(H2_trace{ll,1}(:,1));
    H2_trace{ll,1}(:,1) = (abs(H2_trace{ll,1}(:,1) - m)/abs(m) < 3).*H2_trace{ll,1}(:,1);

    for i = 1:7      
        Theta_trace_1N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))));
        Theta_trace_2N{ll,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H2_trace{ll,1}(:,1)) / sum(exp(H2_trace{ll,1}(:,1))));
        Theta_traceN{ll+1,1}(:,i) = Theta_trace{ll+1,1}(:,i) .* (exp(H1_trace{ll,1}(:,1)) / sum(exp(H1_trace{ll,1}(:,1))) - exp(H2_trace{ll,1}(:,1))/ sum(exp(H2_trace{ll,1}(:,1))));
        ML_Theta_trace{ll+1,1}(:,i) = cumsum(Theta_traceN{ll+1, 1}(:,i)) ./ (1:Nl(ll+1))';
    end
end

%Theta_traceN{1,1} = mean(Theta_trace{1,1}(burnin:end,:));
Theta_traceN{1,1} = Theta_trace{1,1}(burnin:end,:);

final_theta = Theta_traceN{1,1};
level_means = zeros(LP-Lmin, 7);

for i=1:7
    for j = 1:LP - Lmin
        final_theta(i) = final_theta(i) + sum(Theta_traceN{j+1,1}(burnin:end,i));
        level_means(j,i) = sum(Theta_traceN{j+1,1}(burnin:end,i));
    end
end


burnin = 100;
figure
plot(burnin:Nl(1), Theta_trace{2,1}(burnin:end,1), 'r--');
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

for l = 1:LP-Lmin
figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,1), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off
figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,2), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off
figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,3), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off

figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,4), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off
figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,5), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off
figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,6), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off

figure
plot(burnin:Nl(l+1), Theta_traceN{l+1,1}(burnin:end,7), 'r--')
title(['MLPMMH increments l = ', num2str(l+1)])
%legend('L','theta')
hold off
end



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



function [lwf, lwc, cz] = c_pf_db(L, NP, T, T_obs, X,a1,a2,a3,a4,sig1,sig2,rho)

    hlf = 2^(-L);
    hlc = 2^(-(L-1));

    Xf_est = zeros(NP, 2, T);
    Xf_est(:,1,1) = X(1,1) * ones(NP,1);
    Xf_est(:,2,1) = X(2,1) * ones(NP,1);

    Xc_est = zeros(NP,2,T);
    Xc_est(:,1,1) = X(1,1) * ones(NP,1);
    Xc_est(:,2,1) = X(2,1) * ones(NP,1);

  
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

    lwf_vector = zeros(NP,1);
    lwc_vector = zeros(NP,1);
  
     for k = 1:T-2

        XKf = zeros(NP,2);
        XKMf = zeros(NP,2);
        XKc = zeros(NP,2);
        XKMc = zeros(NP,2);
        %update x_{k+1}
        %the first observation is missed
        %simulate x^1_{k+1}conditional on X^2_{k+1} and X_k
        
        XKMf(:,1) = Xf_est(:,1,k);
        XKMf(:,2) = Xf_est(:,2,k);
        XKMc(:,1) = Xc_est(:,1,k);
        XKMc(:,2) = Xc_est(:,2,k);
        steps = T_obs(k+1) - T_obs(k);
        
        nlf = ceil(steps/hlf);
        nlc = ceil(steps/hlc);
       
        %{
        if X(1,k+1) == 0 && X(2,k+1) ~= 0
            Xf_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xf_est(:,1,k+1) = sqrt((1-rho^2)*steps)*SIG1*(0.1)*randn(NP,1) + Xf_est(:,1,k) + rho*SIG1/SIG2* (Xf_est(:,2,k+1)-Xf_est(:,2,k));
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,2,k+1) = X(2,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xc_est(:,1,k+1) = sqrt((1-rho^2)*steps)*SIG1*(0.1)*randn(NP,1) + Xc_est(:,1,k) + rho*SIG1/SIG2* (Xc_est(:,2,k+1)-Xc_est(:,2,k));
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);


            [Xf_Bri,Xc_Bri] = cou_em_b(XKMf, XKf, XKMc,XKc, steps, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion Xf_tk+1 ~ N(Xf_tk,Sigma*Sigma'*1)
            lGLf1(:,k) = lG(XKf,XKMf,Cova*steps);
            %h_hat, conditional density 
            lGLf2(:,k) = -lG(Xf_est(:,1,k+1),Xf_est(:,1,k) + rho*SIG1/SIG2*(Xf_est(:,2,k+1)-Xf_est(:,2,k)),(1-rho^2)*SIG1^2*steps*0.01);            
            % the integral of L
            % if there is any way to replace the for loop
            %
            for m1 = 0:nlf-2
                lGLf3(:,k) = lGLf3(:,k) + hlf*diag((-A*Xf_Bri(:,:,m1+1)')' * Cova_Inv*(XKf-Xf_Bri(:,:,m1+1))'/(steps-m1*hlf));
            end

            lGLf3(:,k) = lGLf3(:,k) + disendf*diag((-A*Xf_Bri(:,:,nlf)')' * Cova_Inv*(XKf-Xf_Bri(:,:,nlf))'/disendf);
            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);

            lGLc1(:,k) = lG(XKc,XKMc,Cova*steps);
            %h_hat, conditional density 
            lGLc2(:,k) = -lG(Xc_est(:,1,k+1),Xc_est(:,1,k) + rho*SIG1/SIG2*(Xc_est(:,2,k+1)-Xc_est(:,2,k)),(1-rho^2)*SIG1^2*steps*0.01);            
            % the integral of L
            % if there is any way to replace the for loop
            %
            for m2 = 0:nlc-2
                lGLc3(:,k) = lGLc3(:,k) + hlc*diag((-A*Xc_Bri(:,:,m2+1)')' * Cova_Inv*(XKc-Xc_Bri(:,:,m2+1))'/(steps-m2*hlc));
            end
            lGLc3(:,k) = lGLc3(:,k) + disendc*diag((-A*Xc_Bri(:,:,nlc)')' * Cova_Inv*(XKc-Xc_Bri(:,:,nlc))'/disendc);
            
            
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
            
            %lGLJ(:,k) = max(lGLf(:,k),lGLc(:,k));
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
        %the second observation is missed
        %simulate x^2_{k+1}conditional on X^1_{k+1} and X_k
        elseif X(2,k+1) == 0 && X(1,k+1) ~= 0
            Xf_est(:,1,k+1) = X(1,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xf_est(:,2,k+1) = sqrt((1-rho^2)*steps)*SIG2*(0.1)*randn(NP,1) + Xf_est(:,2,k) + rho*SIG2/SIG1* (Xf_est(:,1,k+1)-Xf_est(:,1,k));
            XKf(:,1) = Xf_est(:,1,k+1);
            XKf(:,2) = Xf_est(:,2,k+1);

            Xc_est(:,1,k+1) = X(1,k+1) * ones(NP,1);
            %generate the first comnent from f_hat (browanian motion delta_t = 1)
            Xc_est(:,2,k+1) = sqrt((1-rho^2)*steps)*SIG2*(0.1)*randn(NP,1) + Xc_est(:,2,k) + rho*SIG2/SIG1* (Xc_est(:,1,k+1)-Xc_est(:,1,k));
            XKc(:,1) = Xc_est(:,1,k+1);
            XKc(:,2) = Xc_est(:,2,k+1);


            [Xf_Bri,Xc_Bri] = cou_em_b(XKMf, XKf, XKMc, XKc,steps, L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
            %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
            lGLf1(:,k) = lG(XKf,XKMf,Cova*steps);
            %h_hat, conditional density 
            lGLf2(:,k) = -lG(Xf_est(:,2,k+1),Xf_est(:,2,k) + rho*SIG2/SIG1*(Xf_est(:,1,k+1)-Xf_est(:,1,k)),(1-rho^2)*SIG2^2*steps*0.01);            
            % the integral of L
            % if there is any way to replace the for loop
            %
            for m1 = 0:nlf-2
                lGLf3(:,k) = lGLf3(:,k) + hlf*diag((-A*Xf_Bri(:,:,m1+1)')' * Cova_Inv*(XKf-Xf_Bri(:,:,m1+1))'/(steps-m1*hlf));
            end
            lGLf3(:,k) = lGLf3(:,k) + disendf*diag((-A*Xf_Bri(:,:,nlf)')' * Cova_Inv*(XKf-Xf_Bri(:,:,nlf))'/disendf);

            lGLf(:,k) = lGLf3(:,k)+lGLf2(:,k)+lGLf1(:,k);
            lGLc1(:,k) = lG(XKc,XKMc,Cova*steps);
            %h_hat, conditional density 
            lGLc2(:,k) = -lG(Xc_est(:,2,k+1),Xc_est(:,2,k) + rho*SIG2/SIG1*(Xc_est(:,1,k+1)-Xc_est(:,1,k)),(1-rho^2)*steps*SIG2^2*0.01);            
            % the integral of L
            % if there is any way to replace the for loop
            %
            for m2 = 0:nlc-2
                lGLc3(:,k) = lGLc3(:,k) + hlc*diag((-A*Xc_Bri(:,:,m2+1)')' * Cova_Inv*(XKc-Xc_Bri(:,:,m2+1))'/(steps-m2*hlc));
            end
            lGLc3(:,k) = lGLc3(:,k) + disendc*diag((-A*Xc_Bri(:,:,nlc)')' * Cova_Inv*(XKc-Xc_Bri(:,:,nlc))'/(disendc));

           
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);
            
            %lGLJ(:,k) = max(lGLf(:,k), lGLc(:,k));
            lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
        %the second observation is missed
        %simulate x^2_{k+1}conditional on X^1_{k+1} and X_k
        %no missing components
        %}
        if X(k+1,2) ~= 0 && X(k+1,1) ~= 0
            Xf_est(:,1,k+1) = X(k+1,1)*ones(NP,1);
            Xf_est(:,2,k+1) = X(k+1,1)*ones(NP,1);
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
               
            Rfc = Xf_Bri(:,1,1);
            Ffc = Xf_Bri(:,2,1);
            Rfb = Xf_Bri(:,1,1);
            Ffb = Xf_Bri(:,2,1);
            Rfe = XKf(:,1);
            Ffe = XKf(:,2);

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
                    Rfc = Xf_Bri(:,1,m2+2);
                    Ffc = Xf_Bri(:,2,m2+2);
                end
            end
            
            lGLc(:,k) = lGLc3(:,k)+lGLc2(:,k)+lGLc1(:,k);

            %lGLJ(:,k) = max(lGLf(:,k),lGLc(:,k));
            %lGLJ(:,k) = log(0.5*exp(lGLf(:,k)) + 0.5*exp(lGLc(:,k)));
            lGLJ(:,k) = 0.5*lGLf(:,k) + 0.5*lGLc(:,k);

        end
        GL0 = exp(lGLJ(:,k) - max(lGLJ(:,k)));
        lGL_star(1,k)= log(sum(GL0)) + max(lGLJ(:,k));
        lwf = lwf + mean((lGLf(:,k) - lGLJ(:,k)));
        lwc = lwc + mean((lGLc(:,k) - lGLJ(:,k)));

        %lwf_vector = lwf_vector + (lGLf(:,k) - lGLJ(:,k));
        %lwc_vector = lwc_vector + (lGLc(:,k) - lGLJ(:,k));
        

        
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
        %lGLc(:,1:k) = lGLc(I,1:k);
        %lGLJ(:,1:k) = lGLJ(:,1:k);
        %}
     
            
       
     end
     
     %last point
    %update x_{k+1}
    %the two components are missed
    %simulate x_{k+1}conditional on X_k
    %{
    XKf_E = zeros(NP,2);
    XKc_E = zeros(NP,2);
    XKMf_E = zeros(NP,2);
    XKMc_E = zeros(NP,2);

    XKMf_E(:,1) = Xf_est(:,1,T_n1-1);
    XKMf_E(:,2) = Xf_est(:,2,T_n1-1);
    XKMc_E(:,1) = Xc_est(:,1,T_n1-1);
    XKMc_E(:,2) = Xc_est(:,2,T_n1-1);
    
    steps_E = T_obs(T_n1)- T_obs(T_n1 - 1);
    nlfE = ceil(steps_E/hlf);
    nlcE = ceil(steps_E/hlc);
    disendEf = steps_E - (nlfE - 1)*hlf;
    disendEc = steps_E - (nlcE - 1)*hlc;
    RT = chol(eye(2)*steps_E);
    DWT = zeros(NP,2);
    DWT(:,:) = (RT*randn(NP,2)')';

    XKf_E(:,:) = (sqrtm(Cova)*0.9*DWT')' + XKMf_E;
    XKc_E(:,:) = (sqrtm(Cova)*0.9*DWT')' + XKMc_E;
    
    [Xf_Bri_E, Xc_Bri_E]= cou_em_b(XKMf_E, XKf_E, XKMc_E, XKc_E, steps_E,L, NP, A1,A2,A3,A4,SIG1,SIG2,rho);
    %f_tilde from the brownian motion X_tk+1 ~ N(Xtk,Sigma*Sigma'*1)
    lGLf1(:,T_n1-1) = lG(XKf_E,XKMf_E,Cova*steps_E);
    %h_hat, conditional density 
    lGLf2(:,T_n1-1) = -lG(XKf_E,XKMf_E,Cova*steps_E*0.81);
    for m1 = 0:nlfE-2
        lGLf3(:,T_n1-1) = lGLf3(:,T_n1-1) + hlf*diag((-A*Xf_Bri_E(:,:,m1+1)')' * Cova_Inv*(XKf_E-Xf_Bri_E(:,:,m1+1))'/(steps_E-m1*hlf));
    end
    lGLf3(:,T_n1-1) = lGLf3(:,T_n1-1) + disendEf*diag((-A*Xf_Bri_E(:,:,nlfE)')' * Cova_Inv*(XKf_E-Xf_Bri_E(:,:,nlfE))'/disendEf);

    lGLf(:,T_n1-1) = lGLf3(:,T_n1-1)+lGLf2(:,T_n1-1)+lGLf1(:,T_n1-1);

    lGLc1(:,T_n1-1) = lG(XKc_E,XKMc_E,Cova*steps_E);
    %h_hat, conditional density 
    lGLc2(:,T_n1-1) = -lG(XKc_E,XKMc_E,Cova*steps_E*0.81);
    for m2 = 0:nlcE-2
        lGLc3(:,T_n1-1) = lGLc3(:,T_n1-1) + hlc*diag((-A*Xc_Bri_E(:,:,m2+1)')' * Cova_Inv*(XKc_E-Xc_Bri_E(:,:,m2+1))'/(steps_E-m2*hlc));
    end
    lGLc3(:,T_n1-1) = lGLc3(:,T_n1-1) + disendEc*diag((-A*Xc_Bri_E(:,:,nlcE)')' * Cova_Inv*(XKc_E-Xc_Bri_E(:,:,nlcE))'/disendEc);

    lGLc(:,T_n1-1) = lGLc3(:,T_n1-1)+lGLc2(:,T_n1-1)+lGLc1(:,T_n1-1);
    lGLJ(:,T_n1-1) = log(0.5*exp(lGLf(:,T_n1-1)) + 0.5*exp(lGLc(:,T_n1-1)));


    GL0E = exp(lGLJ(:,T_n1-1) - max(lGLJ(:,T_n1-1)));
    lGL_star(1,T_n1-1)= log(sum(GL0E)) + max(lGLJ(:,T_n1-1));

    lwf = lwf + mean((lGLf(:,T_n1-1) - lGLJ(:,T_n1-1)));
    lwc = lwc + mean((lGLc(:,T_n1-1) - lGLJ(:,T_n1-1)));
    
    
    %}
    cz = (T-1) * log(1/NP) + sum(lGL_star);
    %lwf = lwf_vector(I(1));
    %lwc = lwc_vector(I(1));
     
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

function lw = lG(y, x, Cov)
    k = size(x,2);
    lw  = -log(sqrt((2*pi)^k * det(Cov))) - 0.5*diag((y-x) * Cov^(-1) * (y-x)') ;
end


function lpos_p = l_posterior(Theta, lik)
    log_lik = lik;
    log_prior =  lG(Theta(1),-3,3)+lG(Theta(2),-5,3)+lG(Theta(3),-6.5,3)+lG(Theta(4),-4,3)+lG(Theta(5),-2,1)+lG(Theta(6),-2,1)+lG(Theta(7),1,1);
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
