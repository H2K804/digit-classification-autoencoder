%LOADING MNIST DATA

trainImg = fopen('train-images.idx3-ubyte','r','b'); 
% first we have to open the binary file

MagicNumber = fread(trainImg,1,'int32');
% MagicNumber

nImages = fread(trainImg,1,'int32'); % Read the number of images
% disp(nImages);

nRows = fread(trainImg,1,'int32');% Read the number of rows in each image
% disp("Number of rows in eahc image = ");
% disp(nRows);

nCols = fread(trainImg,1,'int32');% Read the number of columns in each image
% nCols

% We can read 16 byte as header information (as previously done) or we can seek to 16th byte of
% file (by fseek command in MATLAB) and then start reading the data of images.
fseek(trainImg,16,'bof');

%initialize number of training images
Ntrain = 5000;

%create empty cell array
MNISTTrainImages = cell(1,Ntrain);

%LABELS
%create zeros array for labels
%LABELS READING
trainlbl = fopen('train-labels.idx1-ubyte','r','b'); % first we have to open the binary file
MagicNumberLabels = fread(trainlbl,1,'int32');

nLabels = fread(trainlbl,1,'int32');% Read the number of labels

%Directly from [2] we know the first label is stored in 8th byte of file in unsigned byte format:
fseek(trainlbl,8,'bof');
MNISTTrainLabels = zeros(10,Ntrain);


%read all images from bytefile
for j = 1:Ntrain
    imgVec = fread(trainImg,28*28,'uchar');
    imgRes = zeros(28,28);
    for i=1:28
        imgRes(i,:)= imgVec((i-1)*28+1:i*28);
    end
    imgNorm = imgRes./255;
    MNISTTrainImages{1,j} = imgNorm;
    
    %reading labels
    imgLabel = fread(trainlbl,1,'uchar');
    labelVec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    if imgLabel == 0
        labelVec(10) = 1;
    else
        labelVec(imgLabel) = 1;
    end
    labelVecT = labelVec.';
    MNISTTrainLabels(:,j) = labelVecT;
    
end

% %check if images are read correctly
% figure();
% clf
% for i = 1:20
%     subplot(4,5,i);
%     imshow(MNISTTrainImages{i});
% end


% MNIST testing dataset reading

%LOADING MNIST Test DATA

testImg = fopen('t10k-images.idx3-ubyte','r','b'); 
% first we have to open the binary file

MagicNumber = fread(testImg,1,'int32');
% MagicNumber

nImages = fread(testImg,1,'int32'); % Read the number of images
% disp(nImages);

nRows = fread(testImg,1,'int32');% Read the number of rows in each image
% disp("Number of rows in eahc image = ");
% disp(nRows);

nCols = fread(testImg,1,'int32');% Read the number of columns in each image
% nCols

% We can read 16 byte as header information (as previously done) or we can seek to 16th byte of
% file (by fseek command in MATLAB) and then start reading the data of images.
fseek(testImg,16,'bof');

%initialize number of test images
Ntest = 5000;

%create empty cell array
MNISTTestImages = cell(1,Ntest);

%LABELS
%create zeros array for labels
%LABELS READING
testlbl = fopen('t10k-labels.idx1-ubyte','r','b'); % first we have to open the binary file
MagicNumberLabels = fread(testlbl,1,'int32');

nLabels = fread(testlbl,1,'int32');% Read the number of labels

%Directly from [2] we know the first label is stored in 8th byte of file in unsigned byte format:
fseek(testlbl,8,'bof');
MNISTTestLabels = zeros(10,Ntest);


%read all images from bytefile
for k = 1:Ntest
    imgVec = fread(testImg,28*28,'uchar');
    imgRes = zeros(28,28);
    for i=1:28
        imgRes(i,:)= imgVec((i-1)*28+1:i*28);
    end
    imgNorm = imgRes./255;
    MNISTTestImages{1,k} = imgNorm;
    
    %reading labels
    imgLabel = fread(testlbl,1,'uchar');
    labelVec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    if imgLabel == 0
        labelVec(10) = 1;
    else
        labelVec(imgLabel) = 1;
    end
    labelVecT = labelVec.';
    MNISTTestLabels(:,k) = labelVecT;
    
end

% %check if images are read correctly
% figure();
% clf
% for i = 1:20
%     subplot(4,5,i);
%     imshow(MNISTTestImages{i});
% end


%% Train Stacked Autoencoders for Image Classification

% [xTrainImages,tTrain] = digitTrainCellArrayData;
xTrainImages = MNISTTrainImages;
tTrain = MNISTTrainLabels;
% [xTestImages,tTest] = digitTestCellArrayData;
xTestImages = MNISTTestImages;
tTest = MNISTTestLabels;

% % Display some of the training images
% clf
% for i = 1:20
%     subplot(4,5,i);
%     imshow(xTrainImages{i});
% end
%% Training the first autoencoder
rng('default')

hiddenSize1 = 100;


autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)

figure()

plotWeights(autoenc1);

feat1 = encode(autoenc1,xTrainImages);

%% Training the second autoencoder

hiddenSize2 = 50;

autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2,feat1);

%% Training the final softmax layer

softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);

view(softnet)

%% Forming a stacked neural network

view(autoenc1)
view(autoenc2)
view(softnet)


deepnet = stack(autoenc1,autoenc2,softnet);

view(deepnet)

imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% Load the test images
% [xTestImages,tTest] = digitTestCellArrayData;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

y = deepnet(xTest);
plotconfusion(tTest,y);


% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(xTest);
plotconfusion(tTest,y);
