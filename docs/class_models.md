**CLASSIFICATION MODELS**

GILA can use two different models:

* Auto models: 
	
	Automatically generates a convolutional network that fits the image sizes and the number of classes
	
	~~~C
	-models auto
	~~~
	
* Pre-trained models:

	Load pretrained models and use GlobalAvgPooling to adapt the pretrained models to the size of the FC layers and numbers of classes
	
	~~~C
	-models {vgg16,vgg19,resnet50...}
	~~~



	
