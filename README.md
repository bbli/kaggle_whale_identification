# kaggle_whale_identification

## Summary
Because there were 5000 classes to classify, a Siamese network would work better here. Now, one issue with a Siamese network is that to make a prediction, I would need to choose a representative from each class, for each of the 10,000 pictures in the test set. Futhermore, a poor choice of a representative can drastically affect the probability(as some pictures barely show the fluke of the whale). So ideally, you would compare with a sample for each classes. But for this dataset, this more or less amount to comparing the entire training set for each test image. 

To combat this, one could implement early termination, where one stops comparing with each classes once we have a comparison that is above say, 85% or more. But the runtime would still be O(n) as only one out of the 5000 classes will return a high probability. So instead, one should somehow smartly sample the labels in such a way that probability of missing the true label from the sample is low.

With that in mind, I figured running a nearest neighbors algorithm on the output feature vectors was what I needed. This way, I could sample only O(log(n)) points(specifically the reduction a k-d tree provides), and if my neural net had properly minimized the contrastive loss, I can be pretty confident a true label point is near my test point. 

Unfortunately, after spending quite a bit of time minimizing the contrastive loss down to my liking, MAP5 percentage(the metric used by the competition) was only at 28%. So still a work in progress

## Repo overview
* Main code is in `Train.py`
    * Model.py, utils.py, DataSet.py are all used by Train.py
* Untitled.ipynb is a jupyter notebook for initial data exploration
