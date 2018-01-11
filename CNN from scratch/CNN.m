%Extract training and testing images from dataset

Train = imageDatastore('train',...
         'IncludeSubfolders',true,'LabelSource','foldernames');
Test = imageDatastore('test',...
         'IncludeSubfolders',true,'LabelSource','foldernames');

Train.countEachLabel
Test.countEachLabel

%Define layers to create the model of CNN
layers = [imageInputLayer([256 256 1])
    convolution2dLayer(3,8,'Padding',1)
    dropoutLayer(0.10)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,16,'Padding',1)
    dropoutLayer(0.10)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
      
options = trainingOptions('sgdm','MaxEpochs',50,...
	'InitialLearnRate',0.0005);

%Train CNN with the training dataset, the model created above and the options defining other parameters of CNN
convnet = trainNetwork(Train,layers,options);

%Classify test images based on the trained CNN.
[YTest, WEIGHTS] = classify(convnet,Test);
Lab_tst = YTest;
TTest = Test.Labels;

abbr_categories={'BED','COR','KIT'};
categories={'bedroom','corridor','kitchen'};

%Calculate recognition accuracy based on the above acquired results of classification
accuracy = sum(YTest == TTest)/numel(TTest)   

%Calculate Confusion Matrix
cm=confusionmat(TTest,YTest)

%Display confusion matrix
fig_handle = figure;
imagesc(cm);