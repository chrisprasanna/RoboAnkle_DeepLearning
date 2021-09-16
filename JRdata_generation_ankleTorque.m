clc
clear
close all

%% Get Data

s5 = ('C:\Users\cpras\Documents\UW\Thesis\Modeling\data\PA_A02_S5\PA_A02_S5_wGRF.mat');
s7 = ('C:\Users\cpras\Documents\UW\Thesis\Modeling\data\PA_A02_S7\PA_A02_S7.mat');
s8 = ('C:\Users\cpras\Documents\UW\Thesis\Modeling\data\PA_A02_S8\PA_A02_S8.mat');
s9 = ('C:\Users\cpras\Documents\UW\Thesis\Modeling\data\PA_A02_S9\PA_A02_S9.mat');

if exist('s5','var')
    filenames = {s5, s7, s8, s9};
    unPoweredTrials = [1, 8, 11, 18]; % [3, 6, 13];
else
    filenames = {s7, s8, s9};
    unPoweredTrials = [3, 6, 13];
end

% numInputs = [1,2,4,5,7,8, 13:18]; % 1:10
numInputs = [1,2,5,8,9,11]; % 1:10

UU = [];
XX = [];
YY = [];
TT = [];

% Get initial estimations from real time NARX network
% Loop through all trials
trialcnt = 1;
for ii = 1:length(filenames)
    
    datafile = filenames{ii};
    
    switch datafile(end-4:end)
        case 'F.mat'
            [time,inputs5,outputs5,GRF_all,stanceIndices,thr,Phi,hsTimes,hsIndices] = LoadOrganizeData(datafile,5,numInputs);
            inputs = inputs5;
            outputs = outputs5;
        case '7.mat'
            [time,inputs7,outputs7,GRF_all,stanceIndices,thr,Phi,hsTimes,hsIndices] = LoadOrganizeData(datafile,7,numInputs);
            inputs = inputs7;
            outputs = outputs7;
        case '8.mat'
            [time,inputs8,outputs8,GRF_all,stanceIndices,thr,Phi,hsTimes,hsIndices] = LoadOrganizeData(datafile,8,numInputs);
            inputs = inputs8;
            outputs = outputs8;
        case '9.mat'
            [time,inputs9,outputs9,GRF_all,stanceIndices,thr,Phi,hsTimes,hsIndices] = LoadOrganizeData(datafile,9,numInputs);
            inputs = inputs9;
            outputs = outputs9;
    end
    
    trials = fields(inputs);
    numTrials = length(trials);
    trialsIndices = 1:numTrials;
    
    for jj = 1:numTrials
        
        x = inputs.(trials{jj});
        t = outputs.(trials{jj});
        
        % x = x';
        % t = t';
        % XX(trialcnt, :, :) = reshape(x, length(t), []);
        % TT(trialcnt, :, :) = reshape(t, length(t), []);
        
        XX(trialcnt, :, :) = x;
        TT(trialcnt, :, :) = t;
        
        trialcnt = trialcnt+1;
    end
end

% Organize cross validation sets
numTrials = size(XX,1);
trialInd = 1:numTrials;
poweredTrials = setdiff(trialInd,unPoweredTrials);


%%

PP = [unPoweredTrials, poweredTrials];

T = TT(PP,:,:);
X = XX(PP,:,:);

numPassive = length(filenames);

passiveFeatures = X(1:numPassive, :, :);
activeFeatures = X(numPassive+1:end, :, :);
passiveResponses = T(1:numPassive, :, :);
activeResponses = T(numPassive+1:end, :, :);

Features = X;
Responses = T;

% TT = orderfields(TT,PP);
% XX = orderfields(XX,PP);
% 
% Data.Features = XX;
% Data.Response = TT;
% 
% orderedTrials = trials(PP);
% passiveTrials = orderedTrials(1:4);
% activeTrials = orderedTrials(5:end);
% 
% for fn = 1:4; Data.Passive.Features.(passiveTrials{fn}) = Data.Features.(passiveTrials{fn}); end
% for fn = 1:length(orderedTrials)-4; Data.Active.Features.(activeTrials{fn}) = Data.Features.(activeTrials{fn}); end
% for fn = 1:4; Data.Passive.Response.(passiveTrials{fn}) = Data.Response.(passiveTrials{fn}); end
% for fn = 1:length(orderedTrials)-4; Data.Active.Response.(activeTrials{fn}) = Data.Response.(activeTrials{fn}); end

%% Save

Data.passiveFeatures = passiveFeatures;
Data.activeFeatures = activeFeatures;
Data.passiveResponses = passiveResponses;
Data.activeResponses = activeResponses;

Data.Features = Features;
Data.Responses = Responses;

save('C:\Users\cpras\Documents\UW\Thesis\Pytorch\Ankle Torque\JR_data_ankleTorque.mat',...
    'Data','-v7')