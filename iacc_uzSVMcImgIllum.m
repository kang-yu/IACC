function iacc_uzSVMcImgIllum
%iacc_uzSVMcImgIllum performs image illumination classification using the
%previously trained SVM model. Training data X uses the concatenated values
%of [R G B] histograms as a matrix format: rows (ie. images) by cloumns of
%[R G B] intersities 256*3.
%
% Running iacc_uzSVMcImgIllum will open the File Explorer to select model
% file and test images. Results will be partly displayed on figure.
% 
%----------------------------
% Kang Yu
% Email: kang.yu@usys.ethz.ch
% 2017-02-08
%
% See also: iacc_trainSVMcImgIllum


%% load and test model

clc; clear

startdir = pwd;

% modFName = 'svmClassifyIllum';
[modFName, PathName] = uigetfile({'*.mat','MAT-files (*.mat)';...
    '*.*', 'All files (*.*)'},...
   'select model file', startdir);

% load saved model
modStr = load(fullfile(PathName, modFName));
modIOName = char(fieldnames(modStr));
cMod = modStr.(modIOName);

%% test folder

% testDir = 'P:\Publications\Publications\in_progress\CC_IP\scripts\test_run_illum_classmodel'
[testFN, tpathname, ~] = uigetfile({'*.*', 'All files (*.*)'},...
    'Select image files', 'MultiSelect', 'on', startdir);

%% prepare test data
% crop image
% [~,testFile] = ff_findFiles(testDir, '.CR2');

nTest = numel(testFN);

yTest = zeros(nTest,1);
xTest = zeros(nTest,256*3);
for i=1:nTest
    imT = imread(fullfile(tpathname, testFN{i}));
    % imT = imcrop(imT, [1 1 5616 2600]);
    imThist = [imhist(imT(:,:,1)); imhist(imT(:,:,2)); imhist(imT(:,:,3))];
    yTest(i) = 1; % dummy label
    xTest(i,:) = imThist';
end

%% plot the first test image

imT1 = imread(fullfile(tpathname,testFN{1}));
% imT1 = imcrop(imT1, [1 1 5616 2600]);
figure;
subplot(221); 
    imshow(imT1)
subplot(223); 
    histogram(imT1(:,:,1)), hold on, 
    histogram(imT1(:,:,2))
    histogram(imT1(:,:,3)), hold off
subplot(122); 
    hp = plot(reshape(xTest(1,:), 256, []));
    hp(1).Color = 'r';
    hp(2).Color = 'g';
    hp(3).Color = 'b';

%% predict lables of test data and plot lables
% tLoss = loss(cMod,xTest,yTest) % yTest dummy label
yTHat = predict(cMod, xTest);

figure
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
for j = 1:nTest;
    if j <= 16
        subplot(4,4,j)
        p = plot(reshape(xTest(j,:), 256, []));
        title(sprintf('Class: %d', yTHat(j)))
        title(sprintf('%s \nPredited Class: %d', testFN{j}, yTHat(j)),'interpreter', 'none')
        p(1).Color = 'r';
        p(2).Color = 'g';
        p(3).Color = 'b';
    else 
        disp('Plot done')
    end
end

%% save predictions

resfp = fullfile(tpathname, [modFName(1:end-4), '_Predicted-Image-Illuminination.txt']);
tbl = table(testFN', yTHat, 'VariableNames',{'Image' 'PredictedLabel'});
writetable(tbl, resfp, 'Delimiter', ',')
winopen(tpathname)

