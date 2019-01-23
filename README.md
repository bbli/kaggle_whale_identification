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
> don't update if 0 loss

* log percentage on each of the top 5
## Thoughts

* Ahh, so the singles have been pushed into the new whale space
    * same label favors new whale by n^2 rather than just n -> too biased
    * divid by standard deviation, b/c everything probably the same

* if greater than 70% a label, make that the first label
* take first layer of resnet

* data augmentation + longer train time
* additive data emphasis feedback at end of each epoch

* test if batch norm after relu works better 
    


* learn data augmentation library -> look at the example on Kaggle
* look at test set distribution

* test if having more fully connected layers help 
    * perhaps dimension is too high
* test whether compare pairwise rather than all n^2 is sufficent
* hypertune margin parameter
* hypertune num of nearest neighbors
    
## What I have learned
* Remember that I don't need x^i to see all pictures, but rather representatives
* all pairwise matters -> wrong code would never improve different loss probably b/c it wasn't pushing the label ball away from other points
* Dimension of output matters -> same loss wouldn't improve until I brought it down
