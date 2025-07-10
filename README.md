![](./FacePrivacy/PRO-Face.png)
# ProFace
ProFace: A trustworthy facial data protection research platform developed by Chongqing University of Posts and Telecommunications (CQUPT). It provides efficient implementations of versatile methods for facial data security analysis and privacy protection developed by CQUPT.

This project consists of several modules: 
- **FacePrivacy:** methods for protecting facial privacy in multiple scenarios.
- **FaceSecurity:** methods for facial data analysis (e.g., DeepFake detection, forensic analysis).
- **WebPortal：** A versitile portal for multimedia security analysis and privacy protection.

## FacePrivacy

![](./FacePrivacy/PRO-Face%20S/assets/faceprivacy.jpg)

This [module](https://github.com/lixionga/ProFace/tree/main/FacePrivacy) implements various algorithms for facial privacy protection.
### Paper List:
**`2025.02`**: `基于妆容风格补丁激活的对抗性人脸隐私保护` **《计算机科学》2025**.
[[paper](https://www.jsjkx.com/CN/10.11896/jsjkx.241200001)][[code](https://github.com/lixionga/ProFace/tree/main/FacePrivacy/Makeup-privacy)]

**`2025.01`**: `iFADIT: Invertible Face Anonymization via Disentangled Identity Transform` **Arxiv**.
[[paper](http://arxiv.org/abs/2501.04390)][[code](https://github.com/lixionga/ProFace/tree/main/FacePrivacy/iFADIT)]

**`2024.04`**: `PRO-Face C: Privacy-Preserving Recognition of Obfuscated Face via Feature Compensation` **IEEE TIFS 2024**.
[[paper](https://ieeexplore.ieee.org/document/10499238)][[code](https://github.com/lixionga/ProFace/tree/main/FacePrivacy/PRO-Face%20C)]

**`2023.11`**: `Invertible Image Obfuscation for Facial Privacy Protection via Secure Flow` **IEEE TCSVT 2023**.
[[paper](https://ieeexplore.ieee.org/document/10366303/)][[code](https://github.com/lixionga/ProFace/tree/main/FacePrivacy/PRO-Face%20S)]

**`2022.10`**: `PRO-Face: A Generic Framework for Privacy-preserving Recognizable Obfuscation of Face Images` **ACM Multimedia 2022**.
[[paper](https://dl.acm.org/doi/10.1145/3503161.3548202)][[code](https://github.com/lixionga/ProFace/tree/main/FacePrivacy/PRO-Face)]

## FaceSecurity

![](./FacePrivacy/PRO-Face%20S/assets/facesecurity.png)

This [module](https://github.com/lixionga/ProFace/tree/main/FaceSecurity)  implements various algorithms for facial data analysis.
### Paper List:
**`2025.`**:`Deepfake Detection Leveraging Self-Blended Artifacts Guided by Facial Embedding Discrepancy` **IEEE TCSVT under review**.[[code](https://github.com/lixionga/ProFace/tree/main/FaceSecurity/EG)]

**`2024.11`**: `Advancing Generalized Deepfake Detector with Forgery Perception Guidance` **ACM Multimedia 2024**.
[[paper](https://doi.org/10.1145/3664647.3680713)][[code](https://github.com/lixionga/ProFace/tree/main/FaceSecurity/FPG)] 

**`2024.08`**: `Inspector for Face Forgery Detection: Defending Against Adversarial Attacks From Coarse to Fine` **IEEE TIP 2024**.
[[paper](https://doi.org/10.1109/TIP.2024.3434388)][[code](https://github.com/lixionga/ProFace/tree/main/FaceSecurity/Inspector)]

## WebPortal

![](./FacePrivacy/PRO-Face%20S/assets/webportal1.png)

This [module](https://github.com/lixionga/ProFace/tree/main/Portal)  implements a trustworthy portal, with all relevant functionalities powered by the algorithms from Face Privacy and Face Security.

This module offers three key functionalities:
### Application Presentation Layer
This portal provides a variety of tools through an online webpage. Depending on the user's needs for face privacy protection or face security analysis, the backend system calls the corresponding models deployed on the server to quickly return the desired results. The face privacy protection module includes functions such as real-time anonymization, offline anonymization, anonymized face recognition, and adversarial privacy protection (e.g., adversarial samples, makeup transfer), which can be applied to scenarios like remote monitoring and online live streaming. The face security analysis module offers two solutions: active defense and passive defense, including four deepfake detection models for image detection, video recognition, and audio-video detection, which are suitable for video conferencing, social media, and other scenarios.

### Function Integration Layer
This portal provides an SDK that integrates various tools, making it convenient to integrate applications across different scenarios and meet diverse business requirements.

### Foundation Code Layer
This portal provides open-source code at the foundational level, enabling developers to gain in-depth understanding and further development capabilities.


