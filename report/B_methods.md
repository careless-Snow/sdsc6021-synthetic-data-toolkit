# B：数据与分类方法草稿

本文可供 D 整合进报告的方法部分。设置仍为草稿，正式实验必须等 G2 冻结；开发试跑结果不能写入正式结果表。

## Data and paired splits

The planned study uses CIFAR-10 with the official ten-class label mapping. For each split seed (0, 1, and 2), 20 examples per class are selected from the official training partition, giving 200 labeled images per split. Selection ranks training indices using a deterministic SHA-256-based key containing the split seed, class ID, and original index. The algorithm is versioned in the source. Original indices, image IDs, class mappings, and file checksums are retained for reproducibility. The five experimental conditions within a seed will share exactly the same real-image split.

For the B-only development check, the same selected 20 images per class are partitioned into 16 training and four development images. Thus each development run trains on 160 images and predicts 40 development images; these are not the formal 20-shot results. No additional labeled examples are introduced for development. The official CIFAR-10 test labels and test performance are not used by the B development workflow.

## Frozen representation and fixed views

Images are represented by a ResNet-18 backbone with torchvision's ImageNet-1K V1 pretrained weights. The classification layer is removed, leaving a 512-dimensional feature vector. Every backbone parameter is frozen; the network remains in evaluation mode under inference mode. Buffer values, including BatchNorm running statistics and counters, are checked before and after extraction.

Each selected image has one base view and three deterministic augmented views. The base view uses bilinear resizing of the short edge to 256 pixels, a 224-pixel center crop, conversion to a tensor, and ImageNet normalization (mean 0.485/0.456/0.406; standard deviation 0.229/0.224/0.225). Augmented views use a 224-pixel random resized crop with scale 0.8–1.0 and aspect ratio 3/4–4/3, followed by horizontal flipping with probability 0.5 and the same normalization. Antialiasing is enabled. Each augmented view's RNG seed is derived from the image ID and one of three fixed view seeds, so views remain identical across conditions and reruns. The development evaluator uses only the base view. Features are currently not L2-normalized.

Cached tensors have shape [images, 4, 512], with explicit image and view IDs. Cache fingerprints include source checksums, split metadata, pretrained-weight checksums, preprocessing settings, implementation identity, and relevant package versions. Corrupt or stale outputs are rejected rather than silently reused.

## Classification and training fairness

Only a ten-class linear head is trained. The current draft uses full-batch SGD, learning rate 0.1, weight decay 0.0001, no momentum, and 200 optimizer updates. These choices were specified before observing development performance and have not been selected using official-test results. They remain provisional until the full G2 check.

The real-only objective is the mean cross-entropy over real images, averaging an image's four view losses first. The implemented mixed-objective helper assigns equal source weights: L = 0.5 mean(L_real) + 0.5 mean(L_synthetic). This keeps the aggregate real-data weight fixed as the retained synthetic count changes. Linear-head initialization and the number of optimizer updates are paired within each seed. The mixed objective is checked with small tensor fixtures; synthetic-data loading and the four mixed-condition CLI runs remain pending A/C hand-off and integration.

## Artifacts and interpretation

Each completed development model stores head weights, initialization, seed, condition, class mapping, backbone reference and weight checksum, preprocessing configuration, feature-cache identity, and training image IDs. Model reloads must reproduce logits exactly before the model is marked successful. Development predictions identify each image, seed, condition, prediction, development label, and configuration hash. Stage logs retain provenance, checksums, timings, cache status, and failures.

The B-only check validates implementation and hand-off structure. It does not complete G1 or G2, establish synthetic-data gains, or replace the planned five-condition, three-seed formal evaluation. The CPU development environment is documented separately from A's GPU environment. Code and draft text were prepared with Codex assistance; member names and final contribution claims must be filled from actual participation.
