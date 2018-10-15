<span style="text-decoration:underline;">**CLASSIFICATION EXAMPLES**<span>


<<span "color:blue">### CIFAR 10<span>

Download the images from: [CIFAR10](https://www.dropbox.com/s/nn2bfybbfj3ly9b/CIFAR10.tgz)

you will find two directories (train, test) and two files (tr.txt, ts.txt)

We can specify the data sources in both ways:

* Directories

~~~shell
python3 gila.py -trdir train -tsdir test -mode class -model auto
~~~

* Files

~~~shell
python3 gila.py -trfile tr.txt -tsfile ts.txt -mode class -model auto
~~~

In both cases GILA will use the training data to train the deep model and the test data as unseen data for error estimation
