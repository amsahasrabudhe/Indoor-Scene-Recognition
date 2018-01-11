% Adapted from MATLAB tutorial on Transfer Learning at:
%https://www.mathworks.com/help/nnet/examples/transfer-learning-using-alexnet.html

%Load dataset
Ds = imageDatastore('dataset1',...
       'IncludeSubfolders',true,'LabelSource','foldernames');

%Split dataset into training and testing samples
[Train,Test] = splitEachLabel(Ds,0.7);
orinet = alexnet;

%Extract Alex Net layers for transfer learning
alexlayers = orinet.Layers(1:end-3);

classes = numel(categories(Train.Labels));

%Create model of CNN using transfer learning from Alex Net denoted by the 'alexlayers'
layers = [
    alexlayers
    fullyConnectedLayer(classes,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

minbSize = 10;
iternum = floor(numel(Train.Labels)/minbSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',minbSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false);

%Train the CNN model with the training dataset and the options defining other parameters of CNN
transfernet = trainNetwork(Train,layers,options);

%Classify test images based on the trained CNN.
YTest = classify(transfernet,Test);
TTest = Test.Labels;

%Calculate Confusion Matrix
Conf = confusionmat(TTest,YTest)

%Calculate recognition accuracy based on the above acquired results of classification
accuracy = mean(YTest == TTest)

%Display confusion matrix
imagesc(Conf)
colorbar