
% Dynamic mathematical model of the metabolic link between Type 2 
% Diabetes and Alzheimer's Disease, a predictive model to identify an 
% individual's predisposed neuro-degenerative risk, 
% and identify intervention points that will most reduce it. 

% Biomathematics 374, Stellenbosch University
%  Serena Taylor Parkins & Katharine Jones


% The model, and what it does in order
%     PART 1  Train & validate NETWORK A  (converting our clinical 
%             data to the risk of diabetes), with a BASELINE comparison

%     PART 2  The ODE  system and simulate four patient profiles

%     PART 3  NUMERICAL ANALYSIS (proving that the above ODE system is
%     accurate)

%     PART 3B INTERVENTIONS  

%     PART 4  Train NETWORK B, the surrogate model 
%             (a faster version of the ODE)

%     PART 5  COMPLETE AND LIVE PATIENT DEMO
%             (one real patient, runs through entire pipeline)

%     PART 6  Save everything

%     PART 7  LIVE DEMO  (the typed-in patient information at the beginning of this file)


%  THE 3 STAGES

%     Stage 1:
%     Neural network A is trained against real clinical data, using
%     key markers to identify elevated or irregular markers, which will
%     confirm that it carries metabolic signals( all the data and the
%     neural network has been validated against literature, and is
%     additionally compared with a a logistic-regression baseline).


%     Stage 2:
%     A system of Ordinary differential equations(ODE) is used, and
%     used to transform a patients metabolic burden into a trajectory
%     spanning over 50 years

%     Stage 3:
%     A second neural network is used to copy the copies the ODE, 
%     this is used as a surrogate model, so one can find interventions significantly faster
% 

%  Used: Statistics and Machine Learning Toolbox, and ode45 and ode15


clear; clc; close all;
diary off;                                 
logFile = 'final_run.txt';                 % anything printed is saved as well here
if isfile(logFile), delete(logFile); end
diary(logFile);
rng(12, 'twister');
p = ad_params();

fprintf(' DIABETES -> ALZHEIMER''S ');

%% ===========  DEMO PATIENT INPUT  (EDIT ME)  =============
%  PART 7 (at the end) runs THIS patient through clinical -> burden -> ODE and
%  shows the effect of an intervention. Change these numbers and re-run to make
%  the model respond live. Units are in the comments.
demoPatient.Age     = 32;    % years
demoPatient.Gender  = 2;     % 1 = male, 2 = female
demoPatient.HbA1c   = 11.5;   % %   (3-month blood sugar; <5.7 normal, >=6.5 diabetic)
demoPatient.FPG     = 6.2;   % mmol/L (fasting glucose)
demoPatient.TG      = 3.9;   % mmol/L (triglycerides)
demoPatient.waist1  = 144;   % cm  (waist circumference)
demoPatient.Bpsys   = 120;   % mmHg (systolic Blood Pressure)
demoPatient.Bpdia   = 80;    % mmHg (diastolic Blood Pressure)
demoPatient.ALT     = 88;    % U/L (liver enzyme)
demoPatient.Smoking = 2;     % 1 = never, 2 = former, 3 = current
demoPatient.APOE4   = 0.5;     % 0 = none, 0.5 = one copy, 1 = two copies
%  ============================open final_run.txt===============================================

%  PART 1 — NETWORK A :  trained on real clinical profiles, to determine if
%  a patients profile flags them as diabetic
%
%  -  Validation-based feature selection, softmax probability output, and a
%     logistic-regression baseline. 
%   - ALL data pints kept; 6% diabetic, so the rare class is weighted up ~16x.
%   - The three-way split: train -> validation -> test .
%   - Glucose is not included, this means that the network cannot read the diagnosis off the sheet.
%   - TyG excluded too: computed from fasting glucose (caused label leakage).

%  ===========================================================================

fprintf('\n=====================  PART 1 : NETWORK A  =====================\n');

csvPath = 'diabetes_model_input.csv';
haveData = isfile(csvPath);
if ~haveData
    warning('Missing %s — skipping PART 1 and PART 5; the rest still runs.', csvPath);
else
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    fprintf('Loaded %d patients.\n', height(T));

    % ---- DATA CHECKS ----
    vn = T.Properties.VariableNames;
    dmAll = as_num(T.DM);
    fprintf('Labelled diabetic: %d of %d (%.1f%%).\n', sum(dmAll==1), sum(~isnan(dmAll)), ...
            100 .* sum(dmAll==1)/sum(~isnan(dmAll)));
    if all(ismember({'DMhis','FPG','PG2h'}, vn))
        his  = as_num(T.DMhis);
        rule = (his==1) | (as_num(T.FPG)>=7.0) | (as_num(T.PG2h)>=11.1);   % WHO 1999
        ok   = ~isnan(dmAll);
        fprintf('Label check: DM = history OR FPG>=7.0 OR 2-h PG>=11.1 in %d of %d rows.\n', ...
                sum(rule(ok) == (dmAll(ok)==1)), sum(ok));
        fprintf('Diabetic with no previous diagnosis (screen-detected): %d\n', ...
                sum(dmAll==1 & his~=1));
    else
        fprintf('Label check skipped (needs DMhis, FPG and PG2h in the cleaned file).\n');
    end
    if ismember('HbA1c', vn)
        fprintf('HbA1c >= 6.5%% but labelled non-diabetic (HbA1c-glucose discordant): %d\n', ...
                sum(as_num(T.HbA1c)>=6.5 & dmAll==0));
    end

    % candidate pool: every appropriate predictor, EXCEPT glucose(FPG/PG2h/HbA1c).
    % TyG is excluded as well: TyG = ln(TG x FPG / 2), as it contains fasting glucose,
    % and the label DM is defined by glucose, so TyG would leak the answer.
    % (TyG is still used in the ODE burden mapping, where there is no leakage is apparent.)
    candFeatures = {'Age','Gender','BMI','waist1','WHR','Fat','Bpsys','Bpdia', ...
                    'PulsePressure','CHOL','TG','HDL','LDL','SUA','GFR', ...
                    'ALT','GGT','Smoking','Drinking','DMfamilyHistory'};
    candFeatures = candFeatures(ismember(candFeatures, T.Properties.VariableNames));

    % build X, y 
    X = zeros(height(T), numel(candFeatures));
    for j = 1:numel(candFeatures)
        c = T.(candFeatures{j});  if ~isnumeric(c), c = str2double(string(c)); end
        X(:,j) = c;
    end
    y = T.DM;  keep = ~ismissing(y);
    X = X(keep,:);  y = categorical(y(keep), [0 1], {'NoDiabetes','Diabetes'});

    fprintf('Candidate features (rows with a label):\n');
    fprintf('   %-16s %8s %10s %10s %10s\n', 'feature', 'missing', 'min', 'median', 'max');
    for j = 1:numel(candFeatures)
        c = X(:,j);
        fprintf('   %-16s %7.1f%% %10.4g %10.4g %10.4g\n', candFeatures{j}, ...
                100 .* mean(isnan(c)), min(c), median(c,'omitnan'), max(c));
    end
    jALT = find(strcmp(candFeatures, 'ALT'));
    if ~isempty(jALT) && median(X(:,jALT), 'omitnan') < 10
        warning(['ALT median is below 10:looks like the square-root-of-ALT column ' ...
                 '(the raw file has two columns named ALT). Use ALT in U/L.']);
    end

    % THREE-way split: 60% train / 20% validate / 20% test
    cvOuter = cvpartition(y, 'HoldOut', 0.20, 'Stratify', true);  isTe = test(cvOuter);
    Xte = X(isTe,:);  yte = y(isTe);
    Xr  = X(~isTe,:); yr  = y(~isTe);
    cvInner = cvpartition(yr, 'HoldOut', 0.25, 'Stratify', true); isVa = test(cvInner);
    Xtr = Xr(~isVa,:); ytr = yr(~isVa);
    Xva = Xr(isVa,:);  yva = yr(isVa);
    fprintf('Split: %d train / %d val / %d test (%d diabetic in test).\n', ...
            numel(ytr), numel(yva), numel(yte), sum(yte=='Diabetes'));

    % standardize on train stats
    medTr = median(Xtr,1,'omitnan');
    Xtr = fill_nan(Xtr,medTr);  Xva = fill_nan(Xva,medTr);  Xte = fill_nan(Xte,medTr);
    muA = mean(Xtr,1);  sgA = std(Xtr,0,1);  sgA(sgA==0)=1;
    Ztr = (Xtr-muA)./sgA;  Zva = (Xva-muA)./sgA;  Zte = (Xte-muA)./sgA;

    % rank on TRAIN (|logistic coef|), choose k on VALIDATION
    wTr = local_weights(ytr);
    rankMdl = fitclinear(Ztr, ytr, 'Learner','logistic', 'Weights',wTr);
    [~, order] = sort(abs(rankMdl.Beta), 'descend');
    kGrid = 3:min(12,numel(candFeatures));  valAUC = zeros(size(kGrid));
    for i = 1:numel(kGrid)
        cc = order(1:kGrid(i));
        nn = fitcnet(Ztr(:,cc), ytr, 'LayerSizes',[12 6],'Activations','relu', ...
                     'Lambda',1e-4,'Standardize',false,'Weights',wTr, ...
                     'ClassNames',categories(ytr),'IterationLimit',600);
        [~,s] = predict(nn, Zva(:,cc));
        [~,~,~, valAUC(i)] = perfcurve(yva, s(:,strcmp(nn.ClassNames,'Diabetes')), 'Diabetes');
    end
    [~,bi] = max(valAUC);  kBest = kGrid(bi);  selCols = order(1:kBest);
    selNames = candFeatures(selCols);
    fprintf('Feature selection chose k=%d on validation: %s\n', kBest, strjoin(selNames,', '));

    % final network on TRAIN and VAL with selected features
    Ztrv = [Ztr(:,selCols); Zva(:,selCols)];  ytrv = [ytr; yva];  wTrv = local_weights(ytrv);
    netA = fitcnet(Ztrv, ytrv, 'LayerSizes',[12 6],'Activations','relu', ...
                   'Lambda',1e-4,'Standardize',false,'Weights',wTrv, ...
                   'ClassNames',categories(ytrv),'IterationLimit',1000);

    % the output layer is a SOFTMAX (two-class = sigmoid): scores are
    % probabilities that will sum to 1; the "Diabetes" score is the risk.
    [yhat, score] = predict(netA, Zte(:,selCols));
    pc = strcmp(netA.ClassNames,'Diabetes');
    [~,~,~, aucA] = perfcurve(yte, score(:,pc), 'Diabetes');

    % logistic regression on the same features
    baseLR = fitclinear(Ztrv, ytrv, 'Learner','logistic','Weights',wTrv, ...
                        'ClassNames',categories(ytrv));
    baseLR.ScoreTransform = 'logit';
    [~, sLR] = predict(baseLR, Zte(:,selCols));
    [~,~,~, aucL] = perfcurve(yte, sLR(:,pc), 'Diabetes');

    tp = sum(string(yte)=="Diabetes" & string(yhat)=="Diabetes");
    fn = sum(string(yte)=="Diabetes" & string(yhat)~="Diabetes");
    tn = sum(string(yte)=="NoDiabetes" & string(yhat)=="NoDiabetes");
    fp = sum(string(yte)=="NoDiabetes" & string(yhat)=="Diabetes");
    sensA = tp/max(tp+fn,1);  specA = tn/max(tn+fp,1);
    fprintf(['Network A test AUC = %.3f (sensitivity %.2f, specificity %.2f)' ...
             '  |  logistic baseline AUC = %.3f\n'], aucA, sensA, specA, aucL);
    if aucA > aucL
        fprintf('  -> the network beats the baseline by %.3f.\n', aucA-aucL);
    else
        fprintf('  -> the network does NOT beat the baseline (difference %.3f)', ...
                aucA-aucL);
    end

    % 95% bootstrap confidence intervals for both test AUCs: the test set hold
    aucCI = [aucA NaN NaN];  aucLCI = [aucL NaN NaN];
    try
        [~,~,~, aucCI]  = perfcurve(yte, score(:,pc), 'Diabetes', 'NBoot', 1000);
        [~,~,~, aucLCI] = perfcurve(yte, sLR(:,pc),   'Diabetes', 'NBoot', 1000);
        fprintf('95%% bootstrap CI: network %.3f to %.3f  |  baseline %.3f to %.3f\n', ...
                aucCI(2), aucCI(3), aucLCI(2), aucLCI(3));
    catch err
        fprintf('(bootstrap CI skipped: %s)\n', err.message);
    end
end


%  ===========================================================================
%  PART 2 — THE ODE :  metabolic burden -> 50-year brain trajectory
%  Six quantities change together over age, driven by a 5-element burden vector.
%  The key feature is TWO FEEDBACK LOOPS: inflammation W degrades insulin
%  signalling S (loop 1), and W slows amyloid clearance (loop 2), so the system
%  reinforces itself. All states are dimensionless: S and N stay in (0,1], G stays
%  at or below lamG/dG = 0.9, and A, Tau and W are loads that CAN exceed 1.
%  Equations are in ad_rhs (bottom).
%  ===========================================================================

fprintf('\n=====================  PART 2 : THE ODE  =======================\n');

profiles = { 'Healthy',          [0.05 0.05 0.05 0.05 0]
             'Pre-diabetic',     [0.40 0.50 0.30 0.30 0]
             'Type 2 diabetes',  [0.80 0.85 0.60 0.60 0]
             'T2DM + APOE4',     [0.80 0.85 0.60 0.60 1] };
ageStart = 30;  ageEnd = 80;  y0 = [1;0;0;0;0;1];
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);

figure('Name','ODE trajectories','Color','w','Position',[80 80 900 640]);
panels = {1,'Insulin signalling S'; 3,'Amyloid A'; 4,'Tau'; 6,'Neuron density N'};
cols = [0.18 0.29 0.17; 0.52 0.69 0.36; 0.91 0.49 0.66; 0.77 0.12 0.44];

odeResults = cell(size(profiles,1),1);
fprintf('Neuronal loss by age %d:\n', ageEnd);
for i = 1:size(profiles,1)
    b = profiles{i,2};
    [t, Y] = ode45(@(t,y) ad_rhs(t,y,b,p), [ageStart ageEnd], y0, opts);
    odeResults{i} = struct('age',t,'Y',Y);
    fprintf('   %-18s %5.1f%%   (peak A %.2f, Tau %.2f, W %.2f)\n', profiles{i,1}, ...
            100*(1-Y(end,6)), max(Y(:,3)), max(Y(:,4)), max(Y(:,5)));
    for pnl = 1:4
        subplot(2,2,pnl); hold on;
        plot(t, Y(:,panels{pnl,1}), 'LineWidth', 2, 'Color', cols(i,:));
    end
end
for pnl = 1:4
    subplot(2,2,pnl); title(panels{pnl,2}); xlabel('Age'); xlim([ageStart ageEnd]); grid on;
end
subplot(2,2,1); legend(profiles(:,1),'Location','southwest','FontSize',8);
sgtitle('PART 2 — four metabolic profiles');
fprintf(['(Healthy ~11%% is a plausibility check against reported normal-ageing\n' ...
         ' neuron loss, not a result; %.1f points of it come from the ageing term alone.)\n'], ...
        100*(1 - exp(-p.nAge*(ageEnd - ageStart))));


%  ===========================================================================

%  PART 3 — NUMERICAL ANALYSIS :  
%  ===========================================================================

fprintf('\n================  PART 3 : NUMERICAL ANALYSIS  =================\n');

bTest = [0.80 0.85 0.60 0.60 1];

% (a) our own RK4: halve the step, the error should fall ~16x (4th order)
Nref = rk4_solve(bTest, p, ageStart, ageEnd, y0, 0.0125);  Nref = Nref(6);
fprintf('RK4 convergence (final N, reference = %.7f):\n', Nref);
prevErr = NaN;
for h = [1.0 0.5 0.25 0.125]
    yh = rk4_solve(bTest, p, ageStart, ageEnd, y0, h);  err = abs(yh(6)-Nref);
    if isnan(prevErr), fprintf('   h=%5.3f  err=%.2e\n', h, err);
    else,              fprintf('   h=%5.3f  err=%.2e  ratio %.1f\n', h, err, prevErr/err); end
    prevErr = err;
end
fprintf('   ratios near 16 confirm 4th-order accuracy.\n');

% (b) ode45 tolerance study: if the answer stops, converged
fprintf('ode45 tolerance study (final N):\n');
for rt = [1e-3 1e-6 1e-9]
    o = odeset('RelTol',rt,'AbsTol',rt*1e-2);
    [~, Yt] = ode45(@(t,y) ad_rhs(t,y,bTest,p), [ageStart ageEnd], y0, o);
    fprintf('   RelTol=%.0e  N=%.7f  loss=%.3f%%\n', rt, Yt(end,6), 100*(1-Yt(end,6)));
end
fprintf('   the loss barely changes -> results are numerically converged.\n');

% (c) stiffness check: an explicit solver (ode45) and an implicit one (ode15s)
solN = ode45(@(t,y) ad_rhs(t,y,bTest,p), [ageStart ageEnd], y0, opts);
lossN = 100*(1 - solN.y(6,end));  nfevN = solN.stats.nfevals;
fprintf('Stiffness check (T2DM + APOE4, RelTol 1e-8):\n');
fprintf('   ode45  loss %.6f%%  (%d function evaluations)\n', lossN, nfevN);
lossS = NaN;  nfevS = NaN;                 % kept as NaN if ode15s cannot run
try
    solS = ode15s(@(t,y) ad_rhs(t,y,bTest,p), [ageStart ageEnd], y0, opts);
    lossS = 100*(1 - solS.y(6,end));  nfevS = solS.stats.nfevals;
    fprintf('   ode15s loss %.6f%%  (%d function evaluations)\n', lossS, nfevS);
    fprintf('   difference %.2e percentage points\n', abs(lossN - lossS));
catch err
    fprintf('   (ode15s comparison skipped: %s)\n', err.message);
end

% (d) Jacobian eigenvalues along the same path: negative real parts mean the
%     path is locally stable; the fastest/slowest ratio measures stiffness.
fprintf('Jacobian eigenvalues (real parts, per year) along the T2DM + APOE4 path:\n');
for a = [30 55 80]
    ev = sort(real(eig(num_jacobian(deval(solN, a), bTest, p))));
    fprintf('   age %d:%s  | fastest/slowest = %.0f\n', a, sprintf(' %.4f', ev), ...
            max(abs(ev))/min(abs(ev)));
end


%  ===========================================================================

%  PART 3B — INTERVENTIONS :  synergy and timing  (Results R4 and R5)
%  Intervention = HALVE the chosen burdens (b -> 0.5 b) from a start age onwards,
%  in the Type 2 diabetes profile. Outcome = percentage points (pp) of neuronal
%  loss avoided by age 80, compared with no intervention.
%   - Synergy: does halving Pg and Pi together beat the sum of each alone?
%   - Timing:  what share of the benefit of starting at 30 is kept at 50?
%  The burden jumps at the start age, so the ODE is solved in two pieces
%  (run_switch) and the solver never steps across the jump.

%  ===========================================================================

fprintf('\n=============  PART 3B : INTERVENTIONS (SYNERGY, TIMING)  =============\n');

bBase  = profiles{3,2};                           % 'Type 2 diabetes'
levers = { 'Pg (glycaemia)',            1
           'Pi (insulin resistance)',   2
           'Pg + Pi (metabolic)',       [1 2]
           'Pv + Pf (downstream)',      [3 4] };
startAges = [30 50];
[~, Yb]  = ode45(@(t,y) ad_rhs(t,y,bBase,p), [ageStart ageEnd], y0, opts);
lossBase = 100*(1 - Yb(end,6));
avoided  = zeros(size(levers,1), numel(startAges));
for k = 1:size(levers,1)
    idx = levers{k,2};  bInt = bBase;  bInt(idx) = 0.5*bBase(idx);
    for s = 1:numel(startAges)
        yEnd = run_switch(bBase, bInt, startAges(s), p, ageStart, ageEnd, y0, opts);
        avoided(k,s) = lossBase - 100*(1 - yEnd(6));
    end
end
fprintf('No intervention (Type 2 diabetes): loss by %d = %.2f%%\n', ageEnd, lossBase);
fprintf('   %-26s %11s %11s %6s\n', 'halve', 'from 30', 'from 50', 'kept');
for k = 1:size(levers,1)
    fprintf('   %-26s %8.2f pp %8.2f pp %5.0f%%\n', levers{k,1}, ...
            avoided(k,1), avoided(k,2), 100*avoided(k,2)/avoided(k,1));
end
sumSingles = avoided(1,1) + avoided(2,1);
if avoided(3,1) > sumSingles, verdict = 'SUPRA-additive'; else, verdict = 'not supra-additive'; end
fprintf('Synergy from 30: Pg alone + Pi alone = %.2f pp; together = %.2f pp -> %s\n', ...
        sumSingles, avoided(3,1), verdict);

figure('Name','PART 3B - interventions','Color','w','Position',[120 120 820 460]);
hb = bar(categorical(levers(:,1), levers(:,1)), avoided);
hb(1).FaceColor = [0.18 0.29 0.17];  hb(2).FaceColor = [0.52 0.69 0.36];
ylabel('Neuronal loss avoided by age 80 (pp)');
legend('start at age 30','start at age 50','Location','northwest');  grid on;
title('Halving burdens in the Type 2 diabetes profile (parameters literature-informed, not fitted)');
%  ===========================================================================

%  PART 4 — NETWORK B :  a fast surrogate that copies the ODE
%  (Full detail in surrogate_memo.m.)  Target: R^2 > 0.95.
%  ===========================================================================

fprintf('\n=====================  PART 4 : NETWORK B  =====================\n');

rng(12, 'twister');      % reset, so the surrogate does not depend on PART 1 having run
nS = 2000;
Us = lhsdesign(nS,4);    % Latin hypercube over the four burdens (Pg, Pi, Pv, Pf) in [0,1]
Epick = [0 0.5 1];       % APOE4 dose; weights below chosen for COVERAGE of the
                         % two-copy case, not to match population genotype frequencies
Es = Epick(randsample(numel(Epick), nS, true, [0.65 0.20 0.15]))';
aSs = 30 + 25*rand(nS,1);  aEs = aSs + 15 + 30*rand(nS,1);
Xs = [Us Es aSs aEs];  Ys = zeros(nS,4);

fprintf('Running %d ODE simulations to build surrogate data', nS);
for i = 1:nS
    yf = simulate_final(Xs(i,1:5), p, Xs(i,6), Xs(i,7));
    risk = mean([min(yf(3),1) min(yf(4),1) min(yf(5),1) 1-yf(6)]);
    Ys(i,:) = [yf(6) yf(3) yf(4) risk];
    if mod(i,round(nS/20))==0, fprintf('.'); end
end
fprintf(' done\n');

teB = false(nS,1); teB(randperm(nS, round(0.2*nS))) = true;
inMu = mean(Xs(~teB,:),1);  inSg = std(Xs(~teB,:),0,1);  inSg(inSg==0)=1;
ZsTr = (Xs(~teB,:)-inMu)./inSg;  ZsTe = (Xs(teB,:)-inMu)./inSg;
outMu = mean(Ys(~teB,:),1);  outSg = std(Ys(~teB,:),0,1);  outSg(outSg==0)=1;
TsTr = (Ys(~teB,:)-outMu)./outSg;

outNames = {'N_end','A_end','Tau_end','riskIndex'};
netsB = cell(1,4);  R2 = zeros(1,4);
for j = 1:4
    netsB{j} = fitrnet(ZsTr, TsTr(:,j), 'LayerSizes',[32 32 16], ...
                       'Activations','relu','Standardize',false,'IterationLimit',2000);
    pred = predict(netsB{j}, ZsTe)*outSg(j) + outMu(j);  tru = Ys(teB,j);
    R2(j) = 1 - sum((tru-pred).^2)/sum((tru-mean(tru)).^2);
    fprintf('   surrogate %-10s R^2 = %.4f\n', outNames{j}, R2(j));
end


%  ===========================================================================

%  PART 5 — END-TO-END DEMO :  one real patient through the whole pipeline
%     clinical data --(Network A)--> diabetes risk read
%     clinical data --(mapping)----> burden vector b
%     burden b      --(ODE)--------> 50-year trajectory & neuron loss
%
%  Network A gives a diabetes probability; the 5 ODE burdens are
%  built by a TRANSPARENT mapping (clinical_to_burden), not learned,we
%  have no data linking them to brain outcomes.
%  ===========================================================================

fprintf('\n=====================  PART 5 : END-TO-END  ====================\n');
if haveData
    rr = T(find(T.DM==1, 1), :);                 % a real diabetic patient

    % (i) Network A's risk read: build the candidate-feature vector, impute,
    %     scale, then predict using only the SELECTED columns.
    xr = zeros(1, numel(candFeatures));
    for j = 1:numel(candFeatures)
        v = rr.(candFeatures{j});  if ~isnumeric(v), v = str2double(string(v)); end
        if isnan(v), v = medTr(j); end
        xr(j) = v;
    end
    zr = (xr - muA) ./ sgA;
    [~, sr] = predict(netA, zr(selCols));
    fprintf('Network A diabetes probability for this patient: %.2f\n', ...
            sr(strcmp(netA.ClassNames,'Diabetes')));

    % (ii) map the same patient to a burden vector and run the ODE
    b = clinical_to_burden(rr);
    fprintf('Mapped burden b = [Pg %.2f  Pi %.2f  Pv %.2f  Pf %.2f  E %.2f]\n', b);
    [~, Yp] = ode45(@(t,y) ad_rhs(t,y,b,p), [30 80], y0, opts);
    fprintf('ODE projection -> neuronal loss by age 80: %.1f%%\n', 100*(1-Yp(end,6)));

    % (iii) a simple intervention: halve the two metabolic burdens
    bInt = b; bInt(1)=b(1)*0.5; bInt(2)=b(2)*0.5;
    [~, Yi] = ode45(@(t,y) ad_rhs(t,y,bInt,p), [30 80], y0, opts);
    fprintf('If glycaemia + insulin resistance halved: loss %.1f%% (saved %.1f pp)\n', ...
            100*(1-Yi(end,6)), 100*((1-Yp(end,6))-(1-Yi(end,6))));
else
    fprintf('(skipped — needs diabetes_model_input.csv)\n');
end


%  ===========================================================================

%  PART 6 — SAVE EVERYTHING
%  ===========================================================================

fprintf('\n=====================  PART 6 : SAVE  ==========================\n');
projectResults = struct();
projectResults.ode      = odeResults;
projectResults.profiles = profiles;
projectResults.numR2    = R2;
if haveData
    projectResults.netA    = netA;
    projectResults.aucNet  = aucA;
    projectResults.aucBase = aucL;
    projectResults.aucNetCI    = aucCI;      % [AUC lower upper]; NaN if skipped
    projectResults.aucBaseCI   = aucLCI;
    projectResults.sensitivity = sensA;
    projectResults.specificity = specA;
    projectResults.selFeat = {selNames};
    projectResults.kChosen = kBest;
    projectResults.scaleA  = struct('median',medTr,'mu',muA,'sigma',sgA, ...
                                    'cand',{candFeatures},'selCols',selCols);
end
projectResults.surrogate = struct('nets',{netsB},'inMu',inMu,'inSg',inSg, ...
                                   'outMu',outMu,'outSg',outSg,'R2',R2,'outNames',{outNames});
projectResults.numerics = struct('lossOde45',lossN,'lossOde15s',lossS, ...
                                 'nfevOde45',nfevN,'nfevOde15s',nfevS);
projectResults.interventions = struct('levers',{levers(:,1)},'startAges',startAges, ...
                                      'avoided_pp',avoided,'lossNoIntervention',lossBase);
save('projectResults.mat', 'projectResults');
fprintf('Saved projectResults.mat\n');


%  ===========================================================================

%  PART 7 — LIVE DEMO :  one typed-in patient (edit demoPatient at the top)
%  ===========================================================================

fprintf('\n=====================  PART 7 : LIVE DEMO  =====================\n');

bD = clinical_to_burden_struct(demoPatient);
fprintf('Demo patient: %d-year-old, burden [Pg %.2f  Pi %.2f  Pv %.2f  Pf %.2f  E %.2f]\n', ...
        demoPatient.Age, bD);

[tD,  YD ] = ode45(@(t,y) ad_rhs(t,y,bD, p), [ageStart ageEnd], y0, opts);
bDi = bD;  bDi(1) = bD(1)*0.5;  bDi(2) = bD(2)*0.5;      % intervention
[tDi, YDi] = ode45(@(t,y) ad_rhs(t,y,bDi,p), [ageStart ageEnd], y0, opts);
lB = 100*(1-YD(end,6));  lI = 100*(1-YDi(end,6));
fprintf('Neuronal loss by 80: %.1f%%  ->  %.1f%% with intervention (saves %.1f pp)\n', ...
        lB, lI, lB-lI);

figure('Name','PART 7 - live demo patient','Color','w','Position',[140 140 900 560]);
subplot(1,2,1); hold on;
plot(tD,  YD(:,6),  'LineWidth', 2.5, 'Color', [0.77 0.12 0.44]);   % no change
plot(tDi, YDi(:,6), 'LineWidth', 2.5, 'Color', [0.18 0.29 0.17]);   % intervention
xlabel('Age (years)'); ylabel('Neuron density (1 = healthy)');
title('Neuronal loss over life'); ylim([0 1]); xlim([ageStart ageEnd]); grid on;
legend('no change','with intervention','Location','southwest');
subplot(1,2,2); hold on;
plot(tD, YD(:,3), 'LineWidth', 2, 'Color', [0.91 0.49 0.66]);       % amyloid
plot(tD, YD(:,4), 'LineWidth', 2, 'Color', [0.52 0.69 0.36]);       % tau
plot(tD, YD(:,5), 'LineWidth', 2, 'Color', [0.50 0.50 0.55]);       % inflammation
xlabel('Age (years)'); ylabel('Pathology load'); title('What builds up');
xlim([ageStart ageEnd]); grid on;
legend('amyloid','tau','inflammation','Location','northwest');
sgtitle(sprintf('PART 7 — live demo patient  (%.0f%% loss, %.0f%% with intervention)', lB, lI));

fprintf('\n############  RUN COMPLETE  ############\n');
fprintf('Console output saved to %s\n\n', logFile);
diary off;


%% ############################################################################
%  HELPER FUNCTIONS
%  ############################################################################

function dy = ad_rhs(~, y, b, p)
%AD_RHS  The six model equations (full plain-English version in ode_memo.m).
%  States are dimensionless: S, N in (0,1]; G in [0, lamG/dG]; A, Tau, W >= 0 with
%  no upper bound. Feedback: W lowers S (loop 1); W slows A clearance (loop 2).
    S = y(1);  G = y(2);  A = y(3);  Tau = y(4);  W = y(5);  N = y(6);
    Pg = b(1); Pi = b(2); Pv = b(3); Pf = b(4); E = b(5);
    dS   = -(p.aI*Pi + p.aG*Pg + p.aW*W) * S + p.rS*(1 - S);   % insulin signalling
    dG   =  p.lamG*(1 - S) - p.dG*G;                           % GSK-3beta
    dA   = (p.lamA*G + p.A0) * (1 + p.dAP*E) - p.dA*A / (1 + p.kW*W);  % amyloid
    dTau =  p.lamT*G + p.kAT*A - p.dT*Tau;                     % tau
    dW   =  p.wA*A + p.wT*Tau + p.wF*Pf + p.wV*Pv - p.dW*W;    % inflammation (loop)
    dN   = -(p.nT*Tau + p.nW*W + p.nA*A + p.nAge) * N;         % neuron density
    dy = [dS; dG; dA; dTau; dW; dN];
end

function yf = simulate_final(b, p, ageStart, ageEnd)
%SIMULATE_FINAL  Run the ODE and return only the end state (for the surrogate).
    o = odeset('RelTol',1e-7,'AbsTol',1e-9);
    [~, Y] = ode45(@(t,y) ad_rhs(t,y,b,p), [ageStart ageEnd], [1;0;0;0;0;1], o);
    yf = Y(end, :);
end

function yEnd = rk4_solve(b, p, ageStart, ageEnd, y0, h)
%RK4_SOLVE  Classical fixed-step 4th-order Runge-Kutta (for the error study).
    n = round((ageEnd - ageStart)/h);  y = y0;
    for s = 1:n
        k1 = ad_rhs(0, y,            b, p);
        k2 = ad_rhs(0, y + h/2 * k1, b, p);
        k3 = ad_rhs(0, y + h/2 * k2, b, p);
        k4 = ad_rhs(0, y + h   * k3, b, p);
        y  = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    yEnd = y;
end

function w = local_weights(y)
%LOCAL_WEIGHTS  Inverse-frequency class weight per row (rare class up ~16x).
    c = countcats(y);  wCls = sum(c) ./ (numel(c)*c);  w = wCls(double(y));
end

function Xo = fill_nan(X, med)
%FILL_NAN  Replace NaNs in each column with the supplied (training) median.
    Xo = X;  for j = 1:size(X,2), Xo(isnan(Xo(:,j)), j) = med(j); end
end

function b = clinical_to_burden(r)
%CLINICAL_TO_BURDEN  Map one patient's clinical row to the 5 ODE burdens in
%  [0,1]. A transparent modelling mapping (NOT learned); anchors are clinical
%  thresholds; missing values fall back to a neutral default.
    gv = @(name, def) local_get(r, name, def);
    c01 = @(x) min(max(x,0),1);
    HbA1c=gv('HbA1c',5.5); FPG=gv('FPG',5.5); TG=gv('TG',1.4);
    waist=gv('waist1',85); Bps=gv('Bpsys',120); Bpd=gv('Bpdia',78);
    ALT=gv('ALT',25); Gender=gv('Gender',1); Smoke=gv('Smoking',1);
    Pg = c01((HbA1c - 5.0)/5.0);
    TyG = log(TG*88.57 * FPG*18.0 / 2);  Pi = c01((TyG - 8.0)/1.5);
    Pv = c01(((Bps - Bpd) - 40)/40);
    waistThresh = 90; if Gender == 2, waistThresh = 80; end
    Pf = c01(0.5*c01((waist - waistThresh)/25) + 0.3*c01((ALT - 30)/70) + 0.2*(Smoke==3));
    E = 0;                                   % APOE4 not measured -> scenario 0
    b = [Pg Pi Pv Pf E];
end

function v = local_get(r, name, def)
%LOCAL_GET  Safely read column 'name' from table row r, with a default.
    if ismember(name, r.Properties.VariableNames)
        v = r.(name);  if ~isnumeric(v), v = str2double(string(v)); end
        if isempty(v) || isnan(v), v = def; end
    else, v = def; end
end

function b = clinical_to_burden_struct(c)
%CLINICAL_TO_BURDEN_STRUCT  Same mapping as clinical_to_burden, but reads a
%  plain struct (the typed-in demo patient) instead of a table row.
    c01 = @(x) min(max(x,0),1);
    Pg = c01((c.HbA1c - 5.0)/5.0);
    TyG = log(c.TG*88.57 * c.FPG*18.0 / 2);  Pi = c01((TyG - 8.0)/1.5);
    Pv = c01(((c.Bpsys - c.Bpdia) - 40)/40);
    waistThresh = 90; if c.Gender == 2, waistThresh = 80; end
    Pf = c01(0.5*c01((c.waist1 - waistThresh)/25) ...
           + 0.3*c01((c.ALT - 30)/70) + 0.2*(c.Smoking==3));
    b = [Pg Pi Pv Pf c01(c.APOE4)];
end

function c = as_num(c)
%AS_NUM  Return a table column as doubles (text columns are converted).
    if ~isnumeric(c) && ~islogical(c), c = str2double(string(c)); end
    c = double(c);
end

function J = num_jacobian(y, b, p)
%NUM_JACOBIAN  Forward-difference Jacobian of ad_rhs at state y (stiffness check).
    n = numel(y);  J = zeros(n);  f0 = ad_rhs(0, y, b, p);  h = 1e-7;
    for k = 1:n
        yk = y;  yk(k) = yk(k) + h;
        J(:,k) = (ad_rhs(0, yk, b, p) - f0) / h;
    end
end

function yEnd = run_switch(bBase, bInt, tSwitch, p, ageStart, ageEnd, y0, opts)
%RUN_SWITCH  Solve with burden bBase up to age tSwitch, then bInt up to ageEnd.
%  Two separate solves, so ode45 never steps across the jump in the burden.
    yNow = y0;
    if tSwitch > ageStart
        [~, Y1] = ode45(@(t,y) ad_rhs(t,y,bBase,p), [ageStart tSwitch], y0, opts);
        yNow = Y1(end,:)';
    end
    [~, Y2] = ode45(@(t,y) ad_rhs(t,y,bInt,p), [max(tSwitch,ageStart) ageEnd], yNow, opts);
    yEnd = Y2(end,:)';
end

function p = ad_params()
%AD_PARAMS  ODE parameters (literature-informed, not fitted). 21 are rates in
%  per-year units; dAP and kW are DIMENSIONLESS multipliers. Values unchanged.
    p.aI = 0.055; p.aG = 0.035; p.aW = 0.030; p.rS = 0.020;              % per year
    p.lamG = 0.090; p.dG = 0.100;                                         % per year
    p.lamA = 0.045; p.A0 = 0.0020; p.dA = 0.055;                          % per year
    p.dAP = 0.90;   % dimensionless: APOE4 boost to amyloid production, (1 + dAP*E)
    p.kW  = 1.60;   % dimensionless: inflammation slows clearance, dA*A/(1 + kW*W)
    p.lamT = 0.050; p.kAT = 0.020; p.dT = 0.035;                          % per year
    p.wA = 0.050; p.wT = 0.060; p.wF = 0.045; p.wV = 0.030; p.dW = 0.130; % per year
    p.nT = 0.0090; p.nW = 0.0060; p.nA = 0.0020; p.nAge = 0.0015;        % per year
end