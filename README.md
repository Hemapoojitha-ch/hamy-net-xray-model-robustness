# hamy-net-xray-model-robustness
Deep learning models for chest X-ray analysis often perform well on clean, curated datasets, but their reliability under the kinds of imperfections common in
clinical imaging remains unclear. In real-world settings, chest X-ray images
can be affected by noise, blur, contrast and exposure variations, and compression artifacts, all of which may degrade model performance. In this work, we
evaluate the robustness of several deep learning architectures, including convolutional and transformer-based models, using controlled distortions applied to the
MIMIC-CXR dataset. We assess model performance across multiple distortion
types and severity levels and examine whether simple training strategies, such as
distortion-aware data augmentation, can improve robustness without reducing accuracy on clean images, providing insights toward developing more dependable
systems for clinical deployment.
