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

> Ahh, so the singles have been pushed into the new whale space
    * same label favors new whale by n^2 rather than just n -> too biased

> I believe same label updates is the reason the random outputs look the same, as the other points get carried along for the ride
    * How to check if code is correct?

> data augmentation + longer train time
> get unique labels
    * also get statistics on prediction

* additive data emphasis feedback at end of each epoch

    


> learn data augmentation library -> look at the example on Kaggle
* look at test set distribution

* test if batch norm after relu works better 
* test if having more fully connected layers help 
    * perhaps dimension is too high
* test whether compare pairwise rather than all n^2 is sufficent
* hypertune margin parameter
* cross entropy?
* take first layer of resnet -> memory issues?
* architecture(since my net only takes one input in)    
* loss weight(pairwise instead of label)
* use that guys phash code
* remove new whale and one image labels

## What I have learned
* Remember that I don't need x^i to see all pictures, but rather representatives
* all pairwise matters -> wrong code would never improve different loss probably b/c it wasn't pushing the label ball away from other points
* Dimension of output matters -> same loss wouldn't improve until I brought it down
