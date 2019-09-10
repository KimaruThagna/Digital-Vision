from imageai.Prediction import ImagePrediction

prediction_obj = ImagePrediction() # prediction class
prediction_obj.setModelTypeAsResNet()
prediction_obj.setModelPath("models/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction_obj.loadModel()
prediction_objs, percentage_probabilities = prediction_obj.predictImage("inputs/img_test.jpg", result_count=3)
# result count gives the number of items we wish to spot in the image. Most common case would be the top 3 items range is 1-1000
for index in range(len(prediction_objs)):
    print(prediction_objs[index] , " : " , percentage_probabilities[index])