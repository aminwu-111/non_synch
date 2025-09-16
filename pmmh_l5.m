LP = 5;

job_id = getenv('SLURM_JOB_ID');
proc_id = getenv('SLURM_PROCID');
folder_read = '';
folder_write = sprintf('%s%s', job_id, '/');
results_filename = sprintf('%sL_%i_%s_%s.txt', folder_write, LP, job_id,proc_id);


rng_seed = sum(clock)*mod(str2num(job_id),10000)*(str2num(proc_id)+1);
rng(rng_seed);


format long
X = readmatrix(sprintf('%s%s', folder_read,'X.txt'));
T_obs = readmatrix(sprintf('%s%s', folder_read, 'T.txt'));


T= length(T_obs)-1;
NP = 40;

%number of iterations for each level
Nl =  floor(3 * 2^(2*LP) - 50);
Theta_trace = zeros(Nl,7);
Theta_A = [-4.3,-7,-6.5,-4,-2,-2,1];
tic;

Theta_A_p = Theta_A;
Theta_SIG_p = [exp(Theta_A_p(1:6)),2/(1+exp(-Theta_A_p(7)))-1];
Z = PF_DB(LP, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));
lZ = Z;
l_pos_Theta_A_p = l_posterior(Theta_A_p, lZ);
Theta_trace(1,:) = Theta_A_p;

N_count_1 = 0;
N_count_last_1 = 0;
N_count_2 = 0;
N_count_last_2 = 0;

Sigma_A1 = 0.35*diag([4,3,3,4]);
Sigma_A2 = 1.5*diag([0.2,0.2,0.5]);

for iter = 1:Nl
 
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
   
    Z_prime_1 = PF_DB(LP, NP, T, T_obs, X, Theta_SIG_prime_1(1),Theta_SIG_prime_1(2),Theta_SIG_prime_1(3),Theta_SIG_prime_1(4),Theta_SIG_prime_1(5),Theta_SIG_prime_1(6),Theta_SIG_prime_1(7));
    lZ_prime_1 = Z_prime_1;
    l_pos_Theta_A_prime_1 = l_posterior(Theta_A_prime_1, lZ_prime_1);
 
    alpha_U1 = min(0, l_pos_Theta_A_prime_1 - l_pos_Theta_A_p);
    U1 = log(rand);
    
    if U1 < alpha_U1
        Theta_A_p = Theta_A_prime_1;
        Theta_SIG_p = Theta_SIG_prime_1;
        lZ = lZ_prime_1;
        l_pos_Theta_A_p = l_pos_Theta_A_prime_1;
        Theta_trace(iter,:) = Theta_A_prime_1; 
        N_count_1 = N_count_1 + 1;        
    else
        Theta_trace(iter,:) = Theta_A_p; 
        lZ = PF_DB(LP, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));
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

    Z_prime_2 = PF_DB(LP, NP, T, T_obs, X, Theta_SIG_prime_2(1),Theta_SIG_prime_2(2),Theta_SIG_prime_2(3),Theta_SIG_prime_2(4),Theta_SIG_prime_2(5),Theta_SIG_prime_2(6),Theta_SIG_prime_2(7));
    lZ_prime_2 = Z_prime_2;
    l_pos_Theta_A_prime_2 = l_posterior(Theta_A_prime_2, lZ_prime_2);
 
    alpha_U2 = min(0, l_pos_Theta_A_prime_2 - l_pos_Theta_A_p);
    U2 = log(rand);
    
    if U2 < alpha_U2
        Theta_A_p = Theta_A_prime_2;
        Theta_SIG_p = Theta_SIG_prime_2;
        lZ = lZ_prime_2;
        l_pos_Theta_A_p = l_pos_Theta_A_prime_2;
        Theta_trace(iter,:) = Theta_A_prime_2; 
        N_count_2 = N_count_2 + 1;        
    else
        Theta_trace(iter,:) = Theta_A_p; 
        lZ = PF_DB(LP, NP, T, T_obs, X, Theta_SIG_p(1),Theta_SIG_p(2),Theta_SIG_p(3),Theta_SIG_p(4),Theta_SIG_p(5),Theta_SIG_p(6),Theta_SIG_p(7));
        l_pos_Theta_A_p2 = l_posterior(Theta_A_p, lZ);  
        if true && (l_pos_Theta_A_p2 ~= -inf)
            l_pos_Theta_A_p = l_pos_Theta_A_p2;
        end        
    end 


end

Aln(1,1) = N_count_1 / Nl(1);
Aln(1,2) = N_count_2 / Nl(1);
toc;

burnin = 1;
final_theta = mean(Theta_trace(burnin:end,:),1);
writematrix(final_theta, results_filename);



%particle filter using diffusion bridge
function z = PF_DB(L, NP, T, T_obs, X, a1,a2,a3,a4,s1,s2,rho)

    hl = 2^(-L);
    X_est = zeros(NP,2,T+1);
    X_est(:,1,1) = X(1,1) * ones(NP,1);
    X_est(:,2,1) = X(1,2) * ones(NP,1);
    lGL1 = zeros(NP,T);
    lGL2 = zeros(NP,T);
    lGL3 = zeros(NP,T);
    lGL = zeros(NP,T);
    lGL_star = zeros(1,T);

    for k = 1:T - 1

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

    XKMf(:,1) = X_est(:,1,T);
    XKMf(:,2) = X_est(:,2,T);
    stepsf = T_obs(T+1)- T_obs(T);
    nlf = ceil(stepsf/hl);
    [sampf, log_denf] = h_log_normal(NP, steps, a1,a2,a3,a4,s1,s2, rho, XKMf(:,1), XKMf(:,2), -1, -1);
    for n = 1:NP 
        X_est(n,1,T+1) = sampf(1,n);
        X_est(n,2,T+1) = sampf(2,n);
        lGL2(n,T) = log_denf(n);
    end
    XKf(:,1) = X_est(:,1,T+1);
    XKf(:,2) = X_est(:,2,T+1);

    [log_densityf,~] = linear_tran_t_aux(NP, 0, stepsf, a1,a2,a3,a4,s1,s2,rho, XKMf(:,1), XKMf(:,2), XKf(:,1), XKf(:,2),XKMf(:,1), XKMf(:,2));       
    for n = 1:NP 
        lGL1(n,k) = log_densityf(n);
    end

    [X_Brif, deri_logxf] = em_b_linear(XKMf, XKf, T_obs(T),T_obs(T+1), L, NP, a1,a2,a3,a4,s1,s2,rho);
               
    Rcf = X_Brif(:,1,1);
    Fcf = X_Brif(:,2,1);
    Rbf = X_Brif(:,1,1);
    Fbf = X_Brif(:,2,1);
    Ref = XKf(:,1);
    Fef = XKf(:,2);
    % the integral of L
    for m = 0:nlf-2 
        dif_muf = zeros(NP,2);
        tcf = m*hl;
        t1f = T_obs(T+1) - T_obs(T);
        for n = 1:NP
            dif_muf(n,:) = [a2*Rcf(n)*(Fbf(n)- Fcf(n) + tcf/t1f*(Fef(n)-Fbf(n))), a3*Fcf(n)*(Rcf(n)- Rbf(n)+ tcf/t1f*(Rbf(n)-Ref(n)))];
            dri_ff = deri_logxf(n,:,m+1);
            lGL3(n,T) = lGL3(n,T) + hl*(dif_muf(n,:)*dri_ff');
        end

        if m ~= nlf-2
            Rcf = X_Brif(:,1,m+2);
            Fcf = X_Brif(:,2,m+2);
        end
    end

      tcef = (nlf-1)*hl;
      hlef = t1f - tcef;
      Rcef = X_Brif(:,1,nlf);
      Fcef = X_Brif(:,2,nlf);
      dif_muef = zeros(NP,2);
      
      for n = 1:NP
          dif_muef(n,:) = [a2*Rcef(n)*(Fbf(n)- Fcef(n) + tcef/t1f*(Fef(n)-Fbf(n))), a3*Fcef(n)*(Rcef(n)- Rbf(n)+ tcef/t1f*(Rbf(n)-Ref(n)))];
          dri_fef = deri_logxf(n,:,nl);
          lGL3(n,T) = lGL3(n,T) + hlef*(dif_muef(n,:)*dri_fef');
      end
    
    lGL(:,T) = lGL3(:,T)+lGL2(:,T)+lGL1(:,T);
    GL0 = exp(lGL(:,T) - max(lGL(:,T)));
    lGL_star(1,T)= log(sum(GL0)) + max(lGL(:,T));
    z = (T - 1) * log(1/NP) + sum(lGL_star);
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


function [samples, log_density] = h_log_normal(N, tdiff, a11,a12, a21,a22, sig1, sig2, rho, R0, F0, Re, Fe)
    samples = zeros(2,N);
    log_density = zeros(1,N);
    if (Re == -1)&& (Fe ~=-1)
        for n = 1:N
            sigma = 0.05;
            mu = log(R0(n)) - sigma^2/2;
            X = lognrnd(mu, sigma);
            samples(1,n) = X;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
            samples(2,n) = Fe;
        end
    end

    if (Fe == -1)&&(Re ~= -1)
        for n = 1:N
            sigma = 0.05;
            mu = log(F0(n)) - sigma^2/2;
            X = lognrnd(mu, sigma);
            samples(2,n) = X;
            log_density(n) = -log(X)+lG(log(X), mu, sigma^2);
            samples(1,n) = Re;
        end
    end
    if (Fe == -1)&&(Re == -1)
        for n = 1:N
            sigma = 0.05;
            mur = log(R0(n)) - sigma^2/2;
            Xr = lognrnd(mur, sigma);
            samples(1,n) = Xr;
            muf = log(F0(n)) - sigma^2/2;
            Xf = lognrnd(muf, sigma);
            samples(2,n) = Xf;
            log_density(n) = -log(Xr)+lG(log(Xr), mur, sigma^2)-log(Xf)+lG(log(Xf), muf, sigma^2);
        
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
