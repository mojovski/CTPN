# CTPN C++

What still needs to be done in order to be able to execute CTPN as a pure c++ program.

## The overall process

1. load image and forward it thrwough the network. The finale layer ist ```text_proposal_layer```, implemented in 
```layers/text_proposal_layer.py```

2. take the output (the text proposals) from the ```text_proposal_layer```, is done in detectors.py:39 or ```detectors.py:21```

3. Filter, cluster and filter again the text proposals.


## The TextProposalLayer

* Uses the anchor generator. ```anchors=self.anchor_generator.locate_anchors((height, width), self._feat_stride)``` in text_proposal_layer.py
	* The method ```locate_anchors``` in the AnchorText class (in ```anchor.py```) calls
		* ```basic_anchors```. which generates all possible anchor sizes (proposals) and calls ```generate_basic_anchors(possible_sizes)```
			* generate_basic_anchors uses the scale_anchor to generate anchors of all possible sizes.
	* generates a 2d-array (grid) and applies some transformations to the anchors using the param layer_params['feat_stride']
* calls ```proposals=self.anchor_generator.apply_deltas_to_anchors(bbox_deltas, anchors)``` where the deltas are the regression result from the trained caffe net. (the method applies a similar approach to SSD detector, where the initial positions are used to predict the class and the boundbox-shift-delta towards the center of the object).
* clip the boxes (simply remove the boxes, which exceed the image size boundaries)


## Filtering, Clustering, etc

For implementation see ```detectors.py:51```




