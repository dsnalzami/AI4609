%% this script are used to classifier. weak learner should be placed here
function [TrainedCLF,CLF_Train_output]=...
    TrainClassifier(TrainIn,TrainTarget,Train_V_Targets,ClassifierType,Params)

switch(ClassifierType)       
    case {'MLPFF'} % MLP       
        CLF = feedforwardnet(Params.MLP.N_nodes); % try change with feedforwardnet or patternnet
        CLF.trainFcn = 'trainlm';            %# training function
        CLF.trainParam.epochs = 1000;        %# max number of iterations
        CLF.trainParam.lr = 0.05;            %# learning rate
        CLF.performFcn = 'mse';              %# mean-squared error function
        CLF.divideFcn = 'dividerand';        %# how to divide data
        CLF = init(CLF);               
        CLF.trainParam.showWindow      = false;
        CLF.trainParam.showCommandLine = false;
        TrainedCLF = train(CLF,TrainIn',Train_V_Targets);
        net_TrainOut = sim(TrainedCLF,TrainIn'); % simulate dynamic system
        
        % produce 3 types of label output
        DP = mapminmax(net_TrainOut',0,1); % decision profile (measurement level output)
        [temp,class_temp] = max(net_TrainOut);  % the abstract level output
        [temp,Ranked_class_temp] = sort(net_TrainOut,'descend'); % rank the level output        
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,net_TrainOut); % Classification confusion matrix
        accuracy =1-TrainError;
        % transpose so we can use it
        class = class_temp';
        Ranked_class = Ranked_class_temp';
        
    case {'LibSVM_RBF'} % LibSVM
        Params.SVM_Params='-t 2 -c 100 -q';
        Params.SVM_Params=[Params.SVM_Params ' -b 1'];
        TrainedCLF = libsvmtrain(TrainTarget,TrainIn,Params.SVM_Params);
        [class,accuracy,prob_estimates] = libsvmpredict(TrainTarget,TrainIn,TrainedCLF,'-q -b 1');
        DP(:,TrainedCLF.Label)=prob_estimates;
        
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
                         
    case {'LibSVM_LINEAR'} % LibSVM
        Params.SVM_Params='-t 0 -c 100 -q';
        Params.SVM_Params=[Params.SVM_Params ' -b 1'];
        TrainedCLF = libsvmtrain(TrainTarget,TrainIn,Params.SVM_Params);
        [class,accuracy,prob_estimates] = libsvmpredict(TrainTarget,TrainIn,TrainedCLF,'-q -b 1');
        DP(:,TrainedCLF.Label)=prob_estimates;
        
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        
    case {'Bayes_N'} % Bayes
        TrainedCLF = fitNaiveBayes(TrainIn,TrainTarget,'Distribution','normal');        
        [class] = predict(TrainedCLF,TrainIn); 
        prob_estimates = posterior(TrainedCLF,TrainIn);
        DP(:,TrainedCLF.ClassLevels)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;
        
     case {'Bayes_K'} % Bayes
        TrainedCLF = fitNaiveBayes(TrainIn,TrainTarget,'Distribution','kernel');        
        [class] = predict(TrainedCLF,TrainIn); 
        prob_estimates = posterior(TrainedCLF,TrainIn);
        DP(:,TrainedCLF.ClassLevels)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;
        
    case {'Bayes_MN'} % Bayes
        TrainedCLF = fitNaiveBayes(TrainIn,TrainTarget,'Distribution','mn');        
        [class] = predict(TrainedCLF,TrainIn); 
        prob_estimates = posterior(TrainedCLF,TrainIn);
        DP(:,TrainedCLF.ClassLevels)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;   
        
    case {'Bayes_MVMN'} % Bayes
        TrainedCLF = fitNaiveBayes(TrainIn,TrainTarget,'Distribution','mvmn');        
        [class] = predict(TrainedCLF,TrainIn); 
        prob_estimates = posterior(TrainedCLF,TrainIn);
        DP(:,TrainedCLF.ClassLevels)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;  
        
    case {'LDA'} % LDA
        TrainedCLF = fitcdiscr(TrainIn,TrainTarget,'DiscrimType','linear');
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;
    
    case {'QDA'} % LDA
        TrainedCLF = fitcdiscr(TrainIn,TrainTarget,'DiscrimType','quadratic');
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;   
    
    case {'DLDA'} % LDA
        TrainedCLF = fitcdiscr(TrainIn,TrainTarget,'DiscrimType','diagLinear');
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;  
        
    case {'DQDA'} % LDA
        TrainedCLF = fitcdiscr(TrainIn,TrainTarget,'DiscrimType','diagQuadratic');
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;    

    case {'PLDA'} % LDA
        TrainedCLF = fitcdiscr(TrainIn,TrainTarget,'DiscrimType','pseudoLinear');
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;    
    
    case {'PQDA'} % LDA
        TrainedCLF = fitcdiscr(TrainIn,TrainTarget,'DiscrimType','pseudoQuadratic');
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;    
        
    case {'DT'} % Decision Tree
        TrainedCLF = fitctree(TrainIn,TrainTarget);
        [class,prob_estimates,node,cnum] = predict(TrainedCLF,TrainIn);        
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;
        
    case {'KNN1'} % KNN
        TrainedCLF = fitcknn(TrainIn,TrainTarget,'NumNeighbors',1,'Distance',Params.KNN.distance);
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;
    case {'KNN2'} % KNN
        TrainedCLF = fitcknn(TrainIn,TrainTarget,'NumNeighbors',2,'Distance',Params.KNN.distance);
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;    
    case {'KNN3'} % KNN
        TrainedCLF = fitcknn(TrainIn,TrainTarget,'NumNeighbors',3,'Distance',Params.KNN.distance);
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;  
    case {'KNN5'} % KNN
        TrainedCLF = fitcknn(TrainIn,TrainTarget,'NumNeighbors',5,'Distance',Params.KNN.distance);
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;    
    case {'KNN10'} % KNN
        TrainedCLF = fitcknn(TrainIn,TrainTarget,'NumNeighbors',10,'Distance',Params.KNN.distance);
        [class,prob_estimates,cost] = predict(TrainedCLF,TrainIn); % score = prob_estimates
        DP(:,TrainedCLF.ClassNames)=prob_estimates;
        [temp,Ranked_class]=sort(DP,2,'descend'); % rank level output
        [TrainError,ConfusionMatrix,CM_ind,CM_per]=confusion(Train_V_Targets,DP');
        accuracy =1-TrainError;   
    case {'MLRC'}
        % due to nature of MLRC, it used as gate for regress, so we dont
        % need use anything in here. check ClassifyTestSamples_Complete2
        %- but we need some as follows:
        TrainedCLF.TrainIn = TrainIn;
        TrainedCLF.TrainTarget = TrainTarget;
        %- we zeroes this parts:
        sizeClass = size(TrainIn,1);
        class = zeros(sizeClass,1);
        Ranked_class = zeros(sizeClass,2);
        DP = Ranked_class;
        accuracy = 0;
        ConfusionMatrix = zeros(2,2);
        CM_per = zeros(2,4);
end

%% train outputs
CLF_Train_output = struct('Abstract_level_output'      , class,...
    'Rank_level_output'          , Ranked_class,...
    'Measurement_level_output'   , DP,...
    'Train_Recognition_rate'     , accuracy,...
    'ConfusionMatrix'            , ConfusionMatrix,...
    'ConfusionMatrix_Percentage' , CM_per);

end
