 _____________________________________________________________________________________________________________________________________________________
|                                                                                                                                                     |
|        ______   _                 __  __                            _    _             __      __           _____     _____   ____    _______       |
|       |  ____| (_)               |  \/  |                          | |  | |     /\     \ \    / /          |  __ \   / ____| |  _ \  |__   __|      |
|       | |__     _   _ __    ___  | \  / |   __ _   _ __    ______  | |  | |    /  \     \ \  / /   ______  | |__) | | |  __  | |_) |    | |         |
|       |  __|   | | | '__|  / _ \ | |\/| |  / _` | | '_ \  |______| | |  | |   / /\ \     \ \/ /   |______| |  _  /  | | |_ | |  _ <     | |         |
|       | |      | | | |    |  __/ | |  | | | (_| | | | | |          | |__| |  / ____ \     \  /             | | \ \  | |__| | | |_) |    | |         |
|       |_|      |_| |_|     \___| |_|  |_|  \__,_| |_| |_|           \____/  /_/    \_\     \/              |_|  \_\  \_____| |____/     |_|         |
|                                                                                                                                                     |
|_____________________________________________________________________________________________________________________________________________________|

Title: FireMan-UAV-RGBT Dataset for Wildfire Detection in Boreal Forests

Authors: 
- S.D.M.W. Kularatne, University of Oulu, manura.sandadurage@oulu.fi
- Kamand Sedaghat Shayegan, University of Oulu, Ksedagha23@student.oulu.fi
- Constantino Álvarez Casado, University of Oulu, constantino.alvarezcasado@oulu.fi (contact person)
- Janne Rajala, University of Oulu, janne.rajala@oulu.fi
- Tuomo Hänninen, University of Oulu, tuomo.hanninen@oulu.fi
- Miguel Bordallo López, University of Oulu, miguel.bordallo@oulu.fi (contact person)
- Le Ngu Nguyen, University of Oulu, le.nguyen@oulu.fi (contact person)


Description:
The FireMan-UAV-RGBT dataset comprises UAV-captured RGB and thermal video data to advance wildfire detection methodologies. Captured over four controlled burnings in Finland during 2022-2023, this dataset includes high-resolution images carefully annotated using manual and semi-automatic methods. The dataset supports advanced drone-based approaches for monitoring and managing wildfires, utilizing both RGB and thermal imaging technologies.

**Content Details:**
- **Videos:** 34 RGB videos, 20 RGB-Thermal pair videos
- **Frames:** Extracted frames from videos
- **Annotations:** Manual and YOLO format annotations for wildfire detection

**Acquisition Details:**
- **Locations:** Ruokolahti, Perho, Kestilä, Taivalkoski (Finland)
- **Dates:** July 6, 2022; August 16, 2022; June 14, 2023; August 26, 2023
- **Equipment:** DJI Matrice 30T and Matrice 300 drones equipped with Zenmuse H20T and H20N cameras

**Annotation Methods:**
- **Manual:** Human annotators drawing bounding boxes and tagging
- **Automatic:** Segmentation models (e.g., SAM, MIL) for thermal images
- **Semi-Automatic:** Projecting thermal annotations to RGB images using image pair registration
- **Tools**: CVAT platform

**Classification Labels:**
- Binary: 'Fire', 'No_Fire'
- Multiclass: 'Fire', 'Smoke', 'Smoke-Fire', 'No-Fire-No-Smoke', 'Weather', 'Vehicle', 'Human', 'Lake', 'Building'

**Dataset Organization:**
- **Training:** 24 Unimodal, 12 Multimodal
- **Validation:** 5 Unimodal, 4 Multimodal
- **Testing:** 5 Unimodal, 4 Multimodal

**Applications:**
This dataset is intended for training and evaluating machine learning models for wildfire detection. It has been tested with deep learning models such as ResNet50 and YOLOv8, showing robust performance in both unimodal and multimodal detection scenarios.

Keywords: wildfire detection, boreal forests, drone videos, RGB-Thermal data, YOLO annotations, machine learning

Funding: This research was funded by Academy of Finland 6G Flagship program under grant number 346208 and Unmanned aerial systems based solutions for real-time management of wildfires under grant number 348008.

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Related/Alternate Identifiers: 
- DOI: 10.5281/zenodo.12628284
- Related publications: Kularatne, S. D. M. W., Álvarez Casado, C., Rajala, J., Hänninen, T., Bordallo López, M., & Nguyen, L. (2024). FireMan-UAV-RGBT: A novel UAV-based RGB-Thermal video dataset for the detection of wildfires in the Finnish forests. In Proceedings of the 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA). IEEE. (Accepted for publication).

Communities: -



