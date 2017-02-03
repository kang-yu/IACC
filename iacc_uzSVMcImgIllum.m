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
% 2017-01-25
%
% See also: iacc_trainSVMcImgIllum


%% load and test model

clear; clc
% startdir = 'O:\FIP\Analysis\2015\WW009\IGB1M\';
startdir = pwd;

% modFName = 'svmClassifyIllum';
[modFName, PathName] = uigetfile({'*.mat','MAT-files (*.mat)';...
    '*.*', 'All files (*.*)'},...
   'select model file', startdir);

% modStr = load([saveDir, modFName]);
modStr = load(fullfile(PathName, modFName));
modIOName = char(fieldnames(modStr));
cMod = modStr.(modIOName);

%% test folder

% testDir = 'P:\Publications\Publications\in_progress\CC_IP\scripts\test_run_illum_classmodel'
[testFN, pathname, ~] = uigetfile({'*.*', 'All files (*.*)'},...
    'Select image files', 'MultiSelect', 'on', startdir);

%% prepare test data
% crop image
% [~,testFile] = ff_findFiles(testDir, '.CR2');

nTest = numel(testFN);

yTest = zeros(nTest,1);
xTest = zeros(nTest,256*3);
for i=1:nTest
    imT = imread(fullfile(pathname, testFN{i}));
    % imT = imcrop(imT, [1 1 5616 2600]);
    imThist = [imhist(imT(:,:,1)); imhist(imT(:,:,2)); imhist(imT(:,:,3))];
    yTest(i) = 1;
    xTest(i,:) = imThist';
end

%% plot the first test image

imT1 = imread(fullfile(pathname,testFN{1}));
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

ooLoss = loss(cMod,xTest,yTest)
yTHat = predict(cMod,xTest);
nVec = 1:size(xTest,1);

figure;
for j = 1:nTest;
    subplot(4,5,j)
    p = plot(reshape(xTest(j,:), 256, [])); 
    title(sprintf('Class: %d',yTHat(j)))
    p(1).Color = 'r';
    p(2).Color = 'g';
    p(3).Color = 'b';
end
