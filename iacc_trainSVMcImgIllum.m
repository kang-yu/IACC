function iacc_trainSVMcImgIllum()
%iacc_trainSVMcImgIllum trains a SVM model for the classification of image
%illumination differences and saves the model for later use on new images.
%
% Usage:
%   put some low light-contrast (LLC) images and hight light-contrast (HLC)
%   images in two folders and call function iacc_trainSVMcImgIllumto select
%   the corresponding images for training. Trained model will be saved to
%   the defined folder. 
% 
%Training data X uses the concatenated intensity values of [R G B] channel
%histograms as a matrix format: rows (ie. images) by cloumns of [R G B]
%intersities 256*3. Training data X is a vector of labels indicating the
%classes/groups of illumination.
%
%
%-----------------------------
% Kang Yu 
% Email: kang.yu@usys.ethz.ch
% 2017-02-08
%
% see also: iacc_trainSVMcImgIllum

clc;clear

startPath = pwd;
saveTime = datestr(now,'yyyymmdd_HHMM');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define model training method
prompt = {'Model Name prefix:',...
    'CV-Method: [1 = holdout, 2 = kfold (default 10)]',...
    'Holdout percent: [0~1]'};
dlg_title = 'Model Training';
num_lines = [1 40; 1 25; 1 20];
defaultans = {'SVMClassifyIllum_', '1', '0.05'};
answer = inputdlg(prompt, dlg_title, num_lines, defaultans);
modfnPref = answer{1};
CV_method = str2double(answer{2});
if isempty(answer{3})
    switch answer{2}
        case '1'
            CV_pn = 0.05;
        case '2'
            CV_pn = 10;
    end
else
    CV_pn = str2double(answer{3});
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LLC img dir

% imFaDir = uigetdir(startPath, 'Select Directory to LLC-Images');
[imfn1, path1, ~] = uigetfile({'*.*', 'All Files (*.*)'}, ...
    'Select LLC-Images', ...
    startPath,...
    'MultiSelect', 'on');

% HLC img dir

% imFbDir = uigetdir(startPath, 'Select Directory to HLC-Images');
[imfn2, path2, ~] = uigetfile({'*.*', 'All Files (*.*)'}, ...
    'Select HLC-Images', ...
    startPath,...
    'MultiSelect', 'on');

% output dir
saveDir = uigetdir(startPath,'select folder to save the model');

%% label LLC images as 1

nFa = numel(imfn1);

yLab1 = zeros(nFa,1);
xDat1 = zeros(nFa,256*3);
for i=1:nFa
    img = imread(fullfile(path1, imfn1{i}));
    img = imcrop(img, [1 1 5616 2600]);
    % Imshow(img)
    imhi = [imhist(img(:,:,1)); imhist(img(:,:,2)); imhist(img(:,:,3))];
    yLab1(i)= 1;
    xDat1(i,:) = imhi';
end

%% label HLC/too-bright images as 2

nFb = numel(imfn2);

yLab2 = zeros(nFb,1);
xDat2 = zeros(nFb,256*3);
for i=1:nFb
    imgb = imread(fullfile(path2, imfn2{i}));
    imgb = imcrop(imgb, [1 1 5616 2600]);
    imhib = [imhist(imgb(:,:,1)); imhist(imgb(:,:,2)); imhist(imgb(:,:,3))];
    yLab2(i) = 2;
    xDat2(i,:) = imhib';
end

%% Plot the 1st observation of each classe

figure

subplot(121), plot(reshape(xDat1(1,:), 256, [])); title(sprintf('Class %d',yLab1(1)));
subplot(122), plot(reshape(xDat2(1,:), 256, [])); title(sprintf('Class %d',yLab2(1)));

%% bind labels of two classes and Xs

xDat = [xDat1; xDat2];
yLab = [yLab1; yLab2];
nObs = numel(yLab);

%% specify the training and holdout samples
% define 25% for testing kfold % modify accordingly

if CV_pn < 1 % holdout
    p = CV_pn;
elseif CV_pn > 1
    p = 0.25;
end

if nObs*p < 1
    p = 1/nObs; % at least allow 1 to be leaved out
    warning('Holdout P is too small. Reset to have meaningful (>=1) number of leave-out samples!')
end

CVP = cvpartition(yLab,'Holdout', p);
inIdx = training(CVP);
ouIdx = test(CVP);                   % Test sample indices
%% train svm model uisng training set

t = templateSVM('SaveSupportVectors',true);

switch CV_method
    case 1
        modSV = fitcecoc(xDat(inIdx,:), yLab(inIdx), 'Learners',t);
        mod = discardSupportVectors(modSV);
        cMod = compact(mod);
        clear mod modSV;
        oosLoss = loss(cMod,xDat(ouIdx,:),yLab(ouIdx))
        yLabHat = predict(cMod,xDat(ouIdx,:));
    case 2
        % NEED TO BE FURTHER DEVELOPED
        
        cvlossthreshold = 0.05; % might be too small when only small number of images used
        if nObs*cvlossthreshold < 1
            cvlossthreshold = 1/nObs; % at least allow 1 to be leaved out
            warning('CVLoss threshold is too small. Reset to have at least 1 wrong classification!')
        end
        
        cvloss = 1;
        while ~(cvloss <= cvlossthreshold)
            cMod = fitcsvm(xDat, yLab, 'KernelFunction','rbf','Standardize',true,'KernelScale','auto');
            cvMod = crossval(cMod);
            cvloss = kfoldLoss(cvMod)
        end
        yLabHat = predict(cMod,xDat(ouIdx,:));
end

%% Assess Holdout Sample Performance

nVec = 1:size(xDat,1);
ouIdx = nVec(ouIdx);
allImNames = [imfn1, imfn2];
ouImNames = allImNames(ouIdx);

figure
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
for j = 1:numel(ouIdx);
    if j <= 16
        subplot(4,4,j)
        plot(reshape(xDat(ouIdx(j),:), 256, []));
        text(0.8, 0.9, sprintf('Label: %d', yLab(ouIdx(j))),'unit','normalized')
        % h = gca;
        title(sprintf('%s \nPredited Class: %d', ouImNames{j}, yLabHat(j)),'interpreter', 'none')
    else
        disp('Plot done')
    end
end

%% save model

modFName = [modfnPref, saveTime];
save([saveDir,'/', modFName,'.mat'], 'cMod');
winopen(saveDir)

