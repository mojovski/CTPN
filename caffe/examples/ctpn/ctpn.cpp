/*This module implements the CTPN python based
in c++

1. subtract the mean, convert to float image

2. reshape and caffe.model.forward2 
in other.py, class CaffeModel:
```
for k, v in input_data.items():
            self.net.blobs[k].reshape(*v.shape)
            self.net.blobs[k].data[...]=v
        return self.net.forward()
```

2.5 AAHH: The prototxt file defines a python layer.
It uses the class ProposalLayer implemented in
src/layers/text_proposals_layer.py


3. read rois and scores: 
```
rois=self.caffe_model.blob("rois")
scores=self.caffe_model.blob("scores")
```

4. takes text proposals having configence >thredhold
```
keep_inds=np.where(scores>cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]
```

5. non maximum supression for each text proposal 
(as stated in the paper, each of width 16px, but different height)
```
keep_inds=nms(np.hstack((text_proposals, scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]
```
nms is the module written in cython from rcnn found in utils/cpu_nms.pyx

6. normalize scores
```
(data-min_)/(max_-min_) if max_-min_!=0 else data-min_
```


7. build text lines from the text proposals.
This is the most difficult part.

7.1 build groups
```
tp_groups=self.group_text_proposals(text_proposals, scores, im_size)
```
uses connected graph method implemented in other.py Graph.sub_graphs_connected(...) class.
But the most tricky part is done in the graph builder in text_proposals:graph_builder.py
method: TextProposalGraphBuilder.build_graph(...)


8. filters boxes.
```
keeP_ids=np.where((widths/heights>cfg.MIN_RATIO) & (scores>cfg.LINE_MIN_SCORE) &
(widths>(cfg.TEXT_PROPOSALS_WIDTH*cfg.MIN_NUM_PROPOSALS)))[0]
text_lines=text_lines[keep_inds]
```


9. NMS for text lines:
```
if text_lines.shape[0]!=0:
    keep_inds=nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
    text_lines=text_lines[keep_inds]
```



*/



#include <caffe/caffe.hpp>
#include "CTPN.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string img_file    = argv[3];
  cv::Vec3f mean(0,0,0);
  CTPN ctpn(model_file, trained_file, mean);


  std::cout << "---------- Text detection for "
            << img_file << " ----------" << std::endl;

  cv::Mat img = cv::imread(img_file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << img_file;
  ctpn.process(img);

}
