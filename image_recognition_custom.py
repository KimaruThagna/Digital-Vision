from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet() #others include SqueezeNet, ResNet, InceptionV3 and DenseNet
prediction.setModelPath("models/idenprof_061-0.7933.h5") # weights as output of the training process on custom dataset
prediction.setJsonPath("idenprof_model_class.json") # mapping
prediction.loadModel(num_objects=10) # prediction classes

predictions, probabilities = prediction.predictImage("inputs/idenprof_image.jpg", result_count=3) # number of possible classes we wish to see
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
