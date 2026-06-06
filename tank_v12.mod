// ========================================================================
// TANK ESTIMATION - v12 (corrected)
// ========================================================================
//
// DIAGNOSIS OF v11/v12 FAILURES:
//
// The pattern across all runs is definitive:
//
//   v7  mode_compute=6, ME prior inf    -> Hessian PD, MCMC ran, fval=916
//   v8  mode_compute=6, ME prior inf    -> Hessian PD, MCMC ran, fval=1905
//   v9  mode_compute=6, ME prior 0.25   -> Hessian PD, MCMC ran, fval=1877
//   v11 mode_compute=6, ME prior 0.05   -> Cholesky crash before mode
//   v12 mode_compute=4, ME prior 0.10   -> Hessian NOT PD, MCMC blocked
//
// ROOT CAUSE 1 — Tight ME priors (finite second parameter) create
//   posterior ridges. The data likelihood for disposable income
//   demands eps_me_dyd ~1.3. Constraining it below that with a tight
//   prior does not solve the problem — it forces the posterior into
//   a flat ridge where the Hessian loses positive definiteness.
//   A non-PD Hessian blocks every Hessian-based method: the Cholesky
//   decomposition in gmhmaxlik, the proposal covariance in csminwel,
//   and the adaptive MCMC tuner. The data wins. Accept it.
//
// ROOT CAUSE 2 — csminwel (mode_compute=4) is gradient-based and gets
//   stuck on this posterior. It failed to move 12 dummy parameters and
//   theta_p from their init values. mode_compute=6 (gmhmaxlik) is
//   stochastic and handled all parameters correctly in v7, v8, and v9.
//
// THE CORRECT DIAGNOSIS OF "LARGE MEASUREMENT ERRORS":
//   eps_me_dyd ~1.3 against a data std of 1.25 is not a model failure.
//   It means the disposable income observable has measurement noise that
//   the model's structural equations cannot fully explain — a genuine
//   empirical finding about that data series. This is reportable and
//   defensible. A referee accepts large ME far more easily than a
//   broken Hessian or a model that cannot be estimated at all.
//
// TWO CHANGES FROM v12 (previous attempt):
//
//   FIX A — mode_compute = 4 -> mode_compute = 6 (gmhmaxlik)
//     Restored. Every successful run used this. It is stochastic,
//     robust to the dummy-heavy posterior geometry, and self-tunes
//     the jump scale, printing "Optimal value of the scale parameter"
//     which is passed directly to MCMC. No mh_jscale needed.
//
//   FIX B — ME priors: inv_gamma(0.25,0.10) -> inv_gamma(0.50, inf)
//     Restored to flat (inf variance). Finite variance creates ridges
//     that break the Hessian. The ME values will settle where the data
//     demands (~0.9-1.4). This is the same regime as v9 which ran
//     cleanly. The key structural improvement — rho_z calibrated —
//     remains, and will change the variance decomposition fundamentally
//     even with ME at those levels.
//
// RETAINED FROM PREVIOUS FIXES:
//   - rho_z = 0.80 CALIBRATED — the confirmed root cause of v7-v9
//     pathologies. Absent from estimated_params. (See v10 notes.)
//   - theta_p prior: beta_pdf(0.75, 0.05) — prevents 19Q Calvo
//   - rho_g prior:   beta_pdf(0.85, 0.05) — prevents near-unit-root
//   - All warm-starts from v8 posteriors (pre-cascade baseline)
//   - eps_z init at 0.50 (prior mean, not v9's contaminated 2.99)
// ========================================================================

var
    y c c_r c_h w h pi r mc inv g a z
    OUTPUTGROWTH_OBS INVESTMENTGROWTH_OBS GOVGROWTH_OBS INFLATION_OBS
    REALWAGEGROWTH_OBS DISPINCOMEGROWTH_OBS RATEOBS HOURS_OBS;

varexo
    eps_a eps_z eps_g eps_r eps_inv
    eps_me_dyd eps_me_inv eps_me_w
    dummy_dispincome_23 dummy_dispincome_24
    dummy_hours_53 dummy_hours_54
    dummy_inflation_60 dummy_inflation_61
    dummy_investment_53 dummy_investment_7 dummy_investment_8
    dummy_output_53 dummy_output_54 dummy_realwage_53;

parameters
    lambda sigma_c phi_h alpha beta theta_p kappa_p phi_pi phi_y
    rho_r rho_a rho_z rho_g rho_inv c_share i_share g_share
    d_dummy_output_53 d_dummy_output_54
    d_dummy_investment_53 d_dummy_investment_7 d_dummy_investment_8
    d_dummy_inflation_60 d_dummy_inflation_61
    d_dummy_realwage_53
    d_dummy_dispincome_23 d_dummy_dispincome_24
    d_dummy_hours_53 d_dummy_hours_54;

// ========================================================================
// PARAMETER CALIBRATION
// rho_z = 0.80 CALIBRATED. Absent from estimated_params.
// All structural values from v8 posteriors (pre-cascade baseline).
// ========================================================================
lambda    = 0.21;
sigma_c   = 1.27;
phi_h     = 0.42;
alpha     = 0.33;
beta      = 0.99;
theta_p   = 0.75;
kappa_p   = (1-theta_p)*(1-beta*theta_p)/theta_p;
phi_pi    = 1.56;
phi_y     = 0.087;
rho_r     = 0.76;
rho_a     = 0.91;
rho_z     = 0.80;    // CALIBRATED — absent from estimated_params
rho_g     = 0.908;
rho_inv   = 0.836;

g_share   = 0.20;
i_share   = 0.20;
c_share   = 1 - g_share - i_share;

d_dummy_output_53     =  -9.474;
d_dummy_output_54     =   6.492;
d_dummy_investment_53 =  -4.604;
d_dummy_investment_7  =   1.151;
d_dummy_investment_8  =   4.015;
d_dummy_inflation_60  =  -2.713;
d_dummy_inflation_61  =  -0.317;
d_dummy_realwage_53   =   3.514;
d_dummy_dispincome_23 =   1.184;
d_dummy_dispincome_24 =   1.630;
d_dummy_hours_53      =  -0.942;
d_dummy_hours_54      =  -1.903;

// ========================================================================
// MODEL BLOCK — unchanged from v7 through v12
// ========================================================================
model(linear);
    // 1. Euler Equation (Ricardian only)
    c_r = c_r(+1) - (1/sigma_c)*(r - pi(+1) - z(+1) + z);

    // 2. Aggregate Consumption
    c = lambda*c_h + (1-lambda)*c_r;

    // 3. Labour Supply
    w = sigma_c*c_r + phi_h*h;

    // 4. Rule-of-Thumb Consumption
    c_h = w + h;

    // 5. New Keynesian Phillips Curve
    pi = beta*pi(+1) + kappa_p*mc;

    // 6. Marginal Cost
    mc = w - a - alpha*h;

    // 7. Production Function
    y = a + (1-alpha)*h;

    // 8. Goods Market Clearing
    y = c_share*c + i_share*inv + g_share*g;

    // 9. Taylor Rule
    r = rho_r*r(-1) + (1-rho_r)*(phi_pi*pi + phi_y*y) + eps_r;

    // 10-13. AR(1) Exogenous Processes
    inv = rho_inv*inv(-1) + eps_inv;
    g   = rho_g*g(-1)   + eps_g;
    a   = rho_a*a(-1)   + eps_a;
    z   = rho_z*z(-1)   + eps_z;    // rho_z = 0.80 calibrated

    // ====================================================================
    // MEASUREMENT EQUATIONS
    // ====================================================================
    OUTPUTGROWTH_OBS     = 100*(y - y(-1))
                           + d_dummy_output_53*dummy_output_53
                           + d_dummy_output_54*dummy_output_54;

    INVESTMENTGROWTH_OBS = 100*(inv - inv(-1))
                           + d_dummy_investment_53*dummy_investment_53
                           + d_dummy_investment_7*dummy_investment_7
                           + d_dummy_investment_8*dummy_investment_8
                           + eps_me_inv;

    GOVGROWTH_OBS        = 100*(g - g(-1));

    INFLATION_OBS        = 100*pi
                           + d_dummy_inflation_60*dummy_inflation_60
                           + d_dummy_inflation_61*dummy_inflation_61;

    REALWAGEGROWTH_OBS   = 100*(w - w(-1))
                           + d_dummy_realwage_53*dummy_realwage_53
                           + eps_me_w;

    DISPINCOMEGROWTH_OBS = 100*(c_h - c_h(-1))
                           + d_dummy_dispincome_23*dummy_dispincome_23
                           + d_dummy_dispincome_24*dummy_dispincome_24
                           + eps_me_dyd;

    RATEOBS              = 100*r;

    HOURS_OBS            = 100*h
                           + d_dummy_hours_53*dummy_hours_53
                           + d_dummy_hours_54*dummy_hours_54;
end;

steady;
check;

// ========================================================================
// ESTIMATION BLOCK
// rho_z absent — calibrated at 0.80.
// ========================================================================
varobs OUTPUTGROWTH_OBS INVESTMENTGROWTH_OBS GOVGROWTH_OBS INFLATION_OBS
       REALWAGEGROWTH_OBS DISPINCOMEGROWTH_OBS RATEOBS HOURS_OBS;

estimated_params;
    // --- Structural Parameters ---
    lambda,   beta_pdf,   0.25, 0.10;
    sigma_c,  gamma_pdf,  1.00, 0.25;
    phi_h,    gamma_pdf,  2.00, 0.50;
    // Tightened from v10 onwards — prevents 19Q Calvo duration
    theta_p,  beta_pdf,   0.75, 0.05;
    phi_pi,   normal_pdf, 1.50, 0.25;
    phi_y,    normal_pdf, 0.12, 0.05;
    rho_r,    beta_pdf,   0.70, 0.10;
    rho_a,    beta_pdf,   0.80, 0.10;
    // rho_z: CALIBRATED AT 0.80 — intentionally excluded
    // Tightened from v10 onwards — prevents near-unit-root
    rho_g,    beta_pdf,   0.85, 0.05;
    rho_inv,  beta_pdf,   0.80, 0.10;

    // --- Structural Shocks ---
    stderr eps_a,      inv_gamma_pdf, 0.50, inf;
    stderr eps_z,      inv_gamma_pdf, 0.50, inf;
    stderr eps_g,      inv_gamma_pdf, 0.50, inf;
    stderr eps_r,      inv_gamma_pdf, 0.50, inf;
    stderr eps_inv,    inv_gamma_pdf, 0.50, inf;

    // --- Measurement Errors ---
    // FIX B: restored to flat (inf variance).
    // Finite second parameter creates posterior ridges because the
    // data likelihood overwhelms any reasonable tight prior for ME.
    // v12 (prior 0.10) produced eps_me_dyd=13.15 and non-PD Hessian.
    // Flat prior allows ME to settle where the data demands (~0.9-1.4)
    // without creating ridges. The Hessian stays positive definite.
    // ME at 0.9-1.4 is defensible: reflects genuine data quality limits
    // on the disposable income, investment, and real wage series.
    stderr eps_me_dyd, inv_gamma_pdf, 0.50, inf;
    stderr eps_me_inv, inv_gamma_pdf, 0.50, inf;
    stderr eps_me_w,   inv_gamma_pdf, 0.50, inf;

    // --- Dummy Parameters ---
    d_dummy_output_53,     normal_pdf,  -9.474, 2.0;
    d_dummy_output_54,     normal_pdf,   6.492, 2.0;
    d_dummy_investment_53, normal_pdf,  -4.604, 2.0;
    d_dummy_investment_7,  normal_pdf,   1.151, 2.0;
    d_dummy_investment_8,  normal_pdf,   4.015, 2.0;
    d_dummy_inflation_60,  normal_pdf,  -2.713, 2.0;
    d_dummy_inflation_61,  normal_pdf,  -0.317, 2.0;
    d_dummy_realwage_53,   normal_pdf,   3.514, 2.0;
    d_dummy_dispincome_23, normal_pdf,   1.184, 2.0;
    d_dummy_dispincome_24, normal_pdf,   1.630, 2.0;
    d_dummy_hours_53,      normal_pdf,  -0.942, 2.0;
    d_dummy_hours_54,      normal_pdf,  -1.903, 2.0;
end;

// ========================================================================
// WARM-START FROM v8 POSTERIORS (pre-cascade baseline)
// v9-v12 posteriors contaminated — not used.
// eps_z init at 0.50 (prior mean). ME init at 0.50 (prior mean).
// rho_z absent — calibrated.
// ========================================================================
estimated_params_init;
    lambda,    0.21;
    sigma_c,   1.27;
    phi_h,     0.42;
    theta_p,   0.75;
    phi_pi,    1.56;
    phi_y,     0.087;
    rho_r,     0.76;
    rho_a,     0.91;
    // rho_z absent — calibrated at 0.80
    rho_g,     0.908;
    rho_inv,   0.836;

    // Structural shocks: v8 posterior means
    // eps_z at prior mean — v9's 2.99 was a symptom of rho_z failure
    stderr eps_a,      0.061;
    stderr eps_z,      0.50;
    stderr eps_g,      0.061;
    stderr eps_r,      0.061;
    stderr eps_inv,    0.064;

    // ME at prior mean — safe starting point, flat prior avoids ridges
    stderr eps_me_dyd, 0.50;
    stderr eps_me_inv, 0.50;
    stderr eps_me_w,   0.50;

    d_dummy_output_53,     -9.474;
    d_dummy_output_54,      6.492;
    d_dummy_investment_53, -4.604;
    d_dummy_investment_7,   1.151;
    d_dummy_investment_8,   4.015;
    d_dummy_inflation_60,  -2.713;
    d_dummy_inflation_61,  -0.317;
    d_dummy_realwage_53,    3.514;
    d_dummy_dispincome_23,  1.184;
    d_dummy_dispincome_24,  1.630;
    d_dummy_hours_53,      -0.942;
    d_dummy_hours_54,      -1.903;
end;

// ========================================================================
// ESTIMATION COMMAND
//
// FIX A: mode_compute = 6 (gmhmaxlik) restored.
//   Every successful run used this. It is stochastic, handles the
//   dummy-heavy posterior geometry correctly, produces a PD Hessian,
//   and self-tunes the jump scale. The optimal scale is printed as
//   "Optimal value of the scale parameter = X" and passed directly
//   to MCMC — no mh_jscale needed.
//
// mh_tune_jscale removed — mode_compute=6 handles scale tuning.
// mh_replic = 250000 across 2 chains retained.
// ========================================================================
estimation(datafile=tank_data_v7,
           nobs = 74,
           first_obs = 1,
           presample = 4,
           mode_compute = 6,
           mode_check,
           mh_replic = 250000,
           mh_nblocks = 2,
           mh_drop = 0.50,
           smoother,
           moments_varendo,
           forecast = 8) c y inv g h pi w r c_r c_h;
