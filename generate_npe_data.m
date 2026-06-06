% =========================================================================
% MASTER THESIS PHASE II: DEFINITIVE NPE SIMULATION ENGINE
% =========================================================================
%
% STRATEGY:
%   1. Dynare compiles the model ONCE via the .mod file
%   2. For each draw: update params → call stoch_simul → extract data
%   3. stoch_simul is the ONLY stable Dynare 6.5 API (resol changed sig)
%
% BUGS FIXED vs. all previous versions:
%   - theta_p prior was Beta(11.25,3.75) [std=0.108] → now Beta(55.5,18.5) [std=0.05]
%   - kappa_p was NEVER recomputed when theta_p changed
%   - Burn-in (100) swallowed the 74-period output → now periods=174, drop=100
%   - Memory hang from invisible plot windows → nograph=1
%   - Package resolution error → rehash path every 5000 draws
%
% =========================================================================

clear all; clc;

%% ========================================================================
%  1. CONFIGURATION
%  ========================================================================
disp('=== PHASE II: NPE Simulation Engine ===');
target_N    = 100000;     % Number of VALID simulations to collect
T_periods   = 74;         % Must match historical dataset length
T_burnin    = 100;        % Dynare-standard burn-in
num_obs     = 8;
num_params  = 18;         % 10 structural + 8 shock std devs

% Paths
script_dir = fileparts(mfilename('fullpath'));
root_dir   = fullfile(script_dir, '..');
mod_dir    = fullfile(root_dir, 'models', 'v12');
data_dir   = fullfile(root_dir, 'data');

% Preallocate output arrays
simulated_params = zeros(target_N, num_params);
simulated_data   = zeros(target_N, num_obs, T_periods);

%% ========================================================================
%  2. INITIALIZE DYNARE (called exactly ONCE)
%  ========================================================================
disp('Compiling model via Dynare...');
cd(mod_dir);

% Force-clean old compiled packages
if exist('+tank_v12_sim', 'dir'); rmdir('+tank_v12_sim', 's'); end
if exist('tank_v12_sim', 'dir');  rmdir('tank_v12_sim', 's');  end
addpath(mod_dir);

dynare tank_v12_sim.mod console noclearall nolog

%% ========================================================================
%  3. PRE-CACHE ALL INDICES (done once, never inside the loop)
%  ========================================================================
disp('Caching variable indices...');

% --- Observable indices (declaration order) ---
obs_names = {'OUTPUTGROWTH_OBS','INVESTMENTGROWTH_OBS','GOVGROWTH_OBS', ...
             'INFLATION_OBS','REALWAGEGROWTH_OBS','DISPINCOMEGROWTH_OBS', ...
             'RATEOBS','HOURS_OBS'};
obs_idx = zeros(1, num_obs);
for i = 1:num_obs
    obs_idx(i) = find(strcmp(obs_names{i}, M_.endo_names));
end

% --- Parameter indices ---
pidx = struct();
pnames = {'lambda','sigma_c','phi_h','theta_p','phi_pi','phi_y', ...
          'rho_r','rho_a','rho_g','rho_inv','kappa_p'};
for i = 1:length(pnames)
    pidx.(pnames{i}) = find(strcmp(pnames{i}, M_.param_names));
end
beta_fixed = M_.params(find(strcmp('beta', M_.param_names)));

% --- Exogenous shock indices ---
exo_names = {'eps_a','eps_z','eps_g','eps_r','eps_inv', ...
             'eps_me_dyd','eps_me_inv','eps_me_w'};
eidx = zeros(1, 8);
for i = 1:8
    eidx(i) = find(strcmp(exo_names{i}, M_.exo_names));
end

nvar = M_.endo_nbr;
nexo = M_.exo_nbr;

fprintf('  Endogenous: %d  |  Exogenous: %d  |  State: %d\n', nvar, nexo, M_.nspred);

%% ========================================================================
%  4. THE SIMULATION LOOP
%  ========================================================================
%
%  Prior Distributions (EXACT from tank_v12_sim.mod estimated_params):
%  -------------------------------------------------------------------
%  lambda   : beta_pdf(0.25, 0.10)  → betarnd(4.4375, 13.3125)
%  sigma_c  : gamma_pdf(1.00, 0.25) → gamrnd(16, 0.0625)
%  phi_h    : gamma_pdf(2.00, 0.50) → gamrnd(16, 0.125)
%  theta_p  : beta_pdf(0.75, 0.05)  → betarnd(55.5, 18.5)   ← TIGHT
%  phi_pi   : normal_pdf(1.50, 0.25)
%  phi_y    : normal_pdf(0.12, 0.05)
%  rho_r    : beta_pdf(0.70, 0.10)  → betarnd(14, 6)
%  rho_a    : beta_pdf(0.80, 0.10)  → betarnd(12, 3)
%  rho_g    : beta_pdf(0.85, 0.05)  → betarnd(42.5, 7.5)    ← TIGHT
%  rho_inv  : beta_pdf(0.80, 0.10)  → betarnd(12, 3)
%  shocks   : inv_gamma(0.50, inf)  → 1/gamrnd(3, 1)
%  -------------------------------------------------------------------

fprintf('\n=== Beginning simulation loop ===\n');
fprintf('Target: %d valid | Burn-in: %d | Output: %d periods\n\n', ...
    target_N, T_burnin, T_periods);

valid_count   = 0;
attempt_count = 0;
tic;

while valid_count < target_N
    attempt_count = attempt_count + 1;
    
    % --- A. Draw from EXACT priors ---
    p_lambda  = betarnd(4.4375, 13.3125);
    p_sigma_c = gamrnd(16, 0.0625);
    p_phi_h   = gamrnd(16, 0.125);
    p_theta_p = betarnd(55.5, 18.5);
    p_phi_pi  = normrnd(1.50, 0.25);
    p_phi_y   = normrnd(0.12, 0.05);
    
    p_rho_r   = betarnd(14, 6);
    p_rho_a   = betarnd(12, 3);
    p_rho_g   = betarnd(42.5, 7.5);
    p_rho_inv = betarnd(12, 3);
    
    s_a      = 1/gamrnd(3, 1);
    s_z      = 1/gamrnd(3, 1);
    s_g      = 1/gamrnd(3, 1);
    s_r      = 1/gamrnd(3, 1);
    s_inv    = 1/gamrnd(3, 1);
    s_me_dyd = 1/gamrnd(3, 1);
    s_me_inv = 1/gamrnd(3, 1);
    s_me_w   = 1/gamrnd(3, 1);
    
    % Quick rejection of obviously invalid draws
    if p_phi_pi < 0 || p_theta_p <= 0.01 || p_theta_p >= 0.99
        continue;
    end
    
    % --- B. Inject into Dynare structures ---
    M_.params(pidx.lambda)  = p_lambda;
    M_.params(pidx.sigma_c) = p_sigma_c;
    M_.params(pidx.phi_h)   = p_phi_h;
    M_.params(pidx.theta_p) = p_theta_p;
    M_.params(pidx.phi_pi)  = p_phi_pi;
    M_.params(pidx.phi_y)   = p_phi_y;
    M_.params(pidx.rho_r)   = p_rho_r;
    M_.params(pidx.rho_a)   = p_rho_a;
    M_.params(pidx.rho_g)   = p_rho_g;
    M_.params(pidx.rho_inv) = p_rho_inv;
    
    % >>> CRITICAL: kappa_p = f(theta_p, beta). Must recompute every draw!
    M_.params(pidx.kappa_p) = (1 - p_theta_p) * (1 - beta_fixed * p_theta_p) / p_theta_p;
    
    % Shock covariance: reset all to zero, then fill 8 active shocks
    M_.Sigma_e = zeros(nexo);
    M_.Sigma_e(eidx(1), eidx(1)) = s_a^2;
    M_.Sigma_e(eidx(2), eidx(2)) = s_z^2;
    M_.Sigma_e(eidx(3), eidx(3)) = s_g^2;
    M_.Sigma_e(eidx(4), eidx(4)) = s_r^2;
    M_.Sigma_e(eidx(5), eidx(5)) = s_inv^2;
    M_.Sigma_e(eidx(6), eidx(6)) = s_me_dyd^2;
    M_.Sigma_e(eidx(7), eidx(7)) = s_me_inv^2;
    M_.Sigma_e(eidx(8), eidx(8)) = s_me_w^2;
    
    % --- C. Solve + Simulate via stoch_simul ---
    % stoch_simul is the ONLY stable public API in Dynare 6.5.
    % resol() changed its signature and crashes with "Not enough input arguments".
    options_.noprint  = 1;
    options_.nomoments = 1;
    options_.irf      = 0;
    options_.nograph   = 1;
    options_.order     = 1;
    options_.periods   = T_periods + T_burnin;   % simulate 174
    options_.drop      = T_burnin;               % drop first 100 → get 74
    
    % Clear previous simulation result
    if isfield(oo_, 'endo_simul')
        oo_ = rmfield(oo_, 'endo_simul');
    end
    
    try
        [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, char());
    catch ME
        if mod(attempt_count, 5000) == 0
            fprintf('  [stoch_simul error @ %d]: %s\n', attempt_count, ME.message);
            rehash path;
        end
        continue;
    end
    
    if info(1) ~= 0
        continue;   % BK violation, indeterminacy, or SS failure
    end
    
    % --- D. Extract simulated data ---
    if ~isfield(oo_, 'endo_simul') || isempty(oo_.endo_simul)
        continue;
    end
    
    sim_data = oo_.endo_simul;
    
    % Extract the 8 observables
    y_obs = sim_data(obs_idx, :);
    
    % Handle variable output sizes: take exactly T_periods columns
    ncols = size(y_obs, 2);
    if ncols < T_periods
        continue;
    elseif ncols > T_periods
        y_obs = y_obs(:, end-T_periods+1:end);
    end
    
    % Reject NaN/Inf (explosive dynamics that slipped past BK)
    if any(isnan(y_obs(:))) || any(isinf(y_obs(:)))
        continue;
    end
    
    % --- E. Store the valid result ---
    valid_count = valid_count + 1;
    
    simulated_params(valid_count, :) = [p_lambda, p_sigma_c, p_phi_h, p_theta_p, ...
                                        p_phi_pi, p_phi_y, p_rho_r, p_rho_a, ...
                                        p_rho_g, p_rho_inv, s_a, s_z, s_g, ...
                                        s_r, s_inv, s_me_dyd, s_me_inv, s_me_w];
    simulated_data(valid_count, :, :) = y_obs;
    
    % --- F. Progress reporting ---
    if valid_count <= 10
        fprintf('  [OK] Sample %d (attempt %d) | lambda=%.3f theta_p=%.3f kappa=%.4f\n', ...
            valid_count, attempt_count, p_lambda, p_theta_p, M_.params(pidx.kappa_p));
    elseif mod(valid_count, 500) == 0
        elapsed = toc;
        rate = valid_count / elapsed;
        eta  = (target_N - valid_count) / rate;
        fprintf('>>> %d / %d valid  |  %d attempts  |  BK pass: %.1f%%  |  ETA: %.0f min\n', ...
            valid_count, target_N, attempt_count, ...
            100*valid_count/attempt_count, eta/60);
    end
    
    % Refresh MATLAB's function cache periodically
    if mod(attempt_count, 5000) == 0
        rehash path;
    end
end

%% ========================================================================
%  5. EXPORT TO DISK
%  ========================================================================
elapsed_total = toc;
fprintf('\n=== Phase II Complete ===\n');
fprintf('  Valid: %d / %d attempts (%.1f%% acceptance)\n', ...
    valid_count, attempt_count, 100*valid_count/attempt_count);
fprintf('  Time: %.1f minutes\n', elapsed_total/60);

if ~exist(data_dir, 'dir'); mkdir(data_dir); end

out_file = fullfile(data_dir, 'npe_training_dataset.mat');
save(out_file, 'simulated_params', 'simulated_data', '-v7.3');

fprintf('  Saved: %s\n', out_file);
fprintf('  simulated_params: [%d x %d]\n', size(simulated_params));
fprintf('  simulated_data:   [%d x %d x %d]\n', size(simulated_data));
fprintf('\nReady for Phase III: PyTorch + sbi.\n');

cd(root_dir);
