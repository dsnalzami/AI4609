%% start the framework
clc;
clear;
% -- start reproducible
%-- random number generator
rng(1986,'twister');

%-- add the path to call all libraries
addpath(genpath(pwd));
%% load feature selection
cd fspackage/
load_fspackage;
cd ../

FS_Type     = 'relieff'; % None|mRMR|chiSquare|fsFisher|relieff|fsCFS|
N_subset    = 20;

ClassifierTypes = 'MLPFF';

Params.MLP.N_nodes  = 10;
% KNN settings
Params.KNN.distance = 'euclidean';
%%
% validation parameters
% Pmethod available method: Kfold | Holdout
Pmethod      = 'Kfold';
N_folds      = 5;
TrainPercent = 0.9;  % Percent of train samples (If P_Method is Holdout)
%
%-- Number of replication
N_runs       = 1;
%
%-- change order of samples in each run(1:yes,0:no)
Reorder      = 1;

% load dataset
dataset = 'E_A_12';
scale = 1;
[Patterns,Targets,V_Targets]=LoadData(dataset,scale);
N_Samples                    = size(Patterns,1);
N_features                   = size(Patterns,2);
N_class                      = size(V_Targets,1);

for run=1:N_runs
    runTime = tic;
    if Reorder==1, y=randperm(N_Samples);
    else y=(1:N_Samples);
    end
    
    Patterns  = Patterns(y,:);
    Targets   = Targets(y);
    V_Targets = V_Targets(:,y);
    
    for fold_N = 1:N_folds
        [TrainPatterns,TrainTargets,Train_V_Targets,TestPatterns,TestTargets,~]=...
            DataPartitioning(Patterns,Targets,V_Targets,Pmethod,fold_N,N_folds,TrainPercent);
        N_test  = length(TestTargets);
        dispCurrK = sprintf('\nYou are in K:%d',fold_N);
        disp(dispCurrK);
        
        % feature selection algorithm
        [RandFeaSelValue,FeatureSubsets] = FeatureSubsetsSelection(FS_Type,N_features,N_subset,N_Samples,TrainPatterns,TrainTargets);
        TrainPatts                       = TrainPatterns(:,FeatureSubsets);
        TestPatts                        = TestPatterns(:,FeatureSubsets);
        N_trainExperiment                = length(TrainPatts);
        
        [TrainedCLF,~]=TrainClassifier(TrainPatts,TrainTargets,Train_V_Targets,ClassifierTypes,Params);
        [Predicted_class,CLF_Test_output]=ClassifyTestSamples(TrainedCLF,TestPatts,TestTargets,ClassifierTypes,Params);
        
        %-- calculate measurements
        % preparing for sensitivity,specificity,precision,f_score, accuracy
        matConf = createConfusionMat(Predicted_class,TestTargets);
        stConf{fold_N}  = confusionmatStats(matConf);
    end % fold_N
    
    stConf2 = stConf';
    stConfRun(run,:) = stConf2;
    
    %--calculations for measurements
    for kfold = 1:N_folds
        stConf3(kfold,1) = stConf2{kfold,1};
    end % end kfold measurements
    
    %--calculations for F-score - other can placed here
    for kfold = 1:N_folds
        MeasResult_Ensemble_se(kfold,1)  = stConf3(kfold).recall(1);
        MeasResult_Ensemble_sp(kfold,1)  = stConf3(kfold).specificity(1);
        MeasResult_Ensemble_pr(kfold,1)  = stConf3(kfold).precision(1);
        MeasResult_Ensemble_f(kfold,1)   = stConf3(kfold).Fscore(1);
        MeasResult_Ensemble_mcc(kfold,1) = stConf3(kfold).mcc(1);
        MeasResult_Ensemble_fpr(kfold,1) = stConf3(kfold).fpr(1);
    end % end kfold f-measurements
    
    % F-measure (macro)
    Measurement_F_Measure_macro(run,:) = mean([stConf3.macroF1]',1);
    
    %-- measurements for one slot (only for first class)
    % sensitivity / recall
    Measurement_sensitivity_Measure_One(run,:) = mean(MeasResult_Ensemble_se);
    % specificity
    Measurement_specificity_Measure_One(run,:) = mean(MeasResult_Ensemble_sp);
    % precision
    Measurement_precision_Measure_One(run,:)   = mean(MeasResult_Ensemble_pr);
    % F-measure
    Measurement_F_Measure_One(run,:)           = mean(MeasResult_Ensemble_f);
    % Matthews correlation coefficient
    Measurement_mcc_Measure_One(run,:)         = mean(MeasResult_Ensemble_mcc);
    % false positive rate (fpr) -> to make ROC = sensitifity and fpr
    Measurement_fpr_Measure_One(run,:)         = mean(MeasResult_Ensemble_fpr);
end % run

disp('The average score of Ensemble System is:');
MeasSensitivity = mean(Measurement_sensitivity_Measure_One,1)
MeasSpecificity = mean(Measurement_specificity_Measure_One,1)
MeasPrecision   = mean(Measurement_precision_Measure_One,1)
MeasFone       = mean(Measurement_F_Measure_One,1)
MeasFmacro     = mean(Measurement_F_Measure_macro,1)
MeasMCC        = mean(Measurement_mcc_Measure_One,1)
MeasFPR       = mean(Measurement_fpr_Measure_One,1)