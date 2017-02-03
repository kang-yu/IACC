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
% 2017-01-24
%
% see also: iacc_trainSVMcImgIllum

% startPath = 'P:\Evaluation\FIP\Analysis';
startPath = pwd;

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

%% define 25% holdout sample and specify the training and holdout samples
% modify accordingly

p = 0.25;
CVP = cvpartition(yLab,'Holdout', p); 
inIdx = training(CVP);               
ouIdx = test(CVP);                   % Test sample indices

%% train svm model uisng training set

t = templateSVM('SaveSupportVectors',true);

% modSV = fitcecoc(xDat, yLab, 'Learners', t, 'KFold', 5);

modSV = fitcecoc(xDat(inIdx,:), yLab(inIdx), 'Learners',t);

%% compact

mod = discardSupportVectors(modSV);
cMod = compact(mod);

clear mod modSV;

%% Assess Holdout Sample Performance

oosLoss = loss(cMod,xDat(ouIdx,:),yLab(ouIdx))
yLabHat = predict(cMod,xDat(ouIdx,:));
nVec = 1:size(xDat,1);
ouIdx = nVec(ouIdx);

figure;
for j = 1:numel(ouIdx);
    subplot(3,3,j)
    plot(reshape(xDat(ouIdx(j),:), 256, []));
    text(0.8, 0.9, sprintf('Label: %d', yLab(ouIdx(j))),'unit','normalized')
    h = gca;
    title(sprintf('Predited Class: %d', yLabHat(j)))
end

%% save model

saveTime = datestr(now,'yyyymmdd_HHMM');

saveDir = uigetdir(startPath,'select folder to save the model');
modFName = ['SVMClassifyIllum_', saveTime];
save([saveDir,'/', modFName,'.mat'], 'cMod');

