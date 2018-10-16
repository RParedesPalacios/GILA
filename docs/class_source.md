<span style="text-decoration:underline;">**CLASSIFICATION DATA SOURCES**<span>

Images can be specified in two different ways:

#### List Files:

A file with the name and class of each image. For example:

~~~
tr.txt:

train/airplane/10008_airplane.png 0
train/automobile/10839_automobile.png 1
train/bird/12362_bird.png 2
train/cat/1316_cat.png 3
train/deer/1297_deer.png 4
train/dog/13121_dog.png 5
train/frog/12885_frog.png 6
train/horse/12633_horse.png 7
train/ship/19097_ship.png 8
train/truck/12979_truck.png 9
~~~

Files can be everywhere, it is not necessary that same class images to be in the same folder. Classes start in 0.

These files will be defined using the "-trfile" and "-tsfile" arguments.

#### Directories:

These method is more convenient because all the images are not loaded in memory but processed in a flow manner.

Each class is a directory containing all the images of that class. In this case same class image **must** be in the same folder. And the folders of the different classes must be in the same directory:

~~~
train/
  -- airpale/
  -- automobile/
  -- bird/
  -- cat/
  -- deer/
  -- dog/
  -- frog/
  -- horse/
  -- ship/
  -- truck/

test/
  -- airpale/
  -- automobile/
  -- bird/
  -- cat/
  -- deer/
  -- dog/
  -- frog/
  -- horse/
  -- ship/
  -- truck/
~~~

These folders will be defined using the "-trdir" and "-tsdir" arguments
