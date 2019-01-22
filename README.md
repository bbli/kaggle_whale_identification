# kaggle_whale_identification

## Data Exploration
* Large imbalence of new whales /lots of whale types with just one sample
    -> Leave in for now
* images of the flank can be small relative to entire image 
    * train specifically these against same label?
    * online community uses bounding boxes for image processing. How easy is it to automate this?
    -> Don't care for now

* images can be of different sizes
    -> Resize for now
##TODO
> test if dataloaders just return 1 -> yes
> check gradient is updating in every for loop
> seperate Losses?
> log percentage of same label targets 
> nearest neighbors at the end
    > just use sklearn nearest neighbor?
    > how to check that my code is correct? Walk through like always
> test whether that guys map5 code works
    * wait until code finishes then just manually calculate first 3

* search for learning rate?

* test if batch norm after relu works better 
* if greater than 70% a label, make that the first label
    


* learn data augmentation library -> look at the example on Kaggle
* look at test set distribution

* test if having more fully connected layers help 
    * perhaps dimension is too high
* test whether compare pairwise rather than all n^2 is sufficent
* hypertune margin parameter
* hypertune num of nearest neighbors
    
## Plan
