# Car Damage Classification and Cost Prediction System 🚗💰

## Table of Contents

* [Introduction](#introduction)
* [Project Objective](#project-objective)
* [Key Functional Modules](#key-functional-modules)
* [Workflow Overview](#workflow-overview)
* [Proposed System Architecture](#proposed-system-architecture)
* [Technologies and Tools Used](#technologies-and-tools-used)
* [Dataset Description](#dataset-description)
* [Employed Models](#employed-models)
* [Novelty](#novelty)
* [Results and Discussion](#results-and-discussion)
* [Conclusion](#conclusion)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Applications](#applications)
* [Future Enhancements](#future-enhancements)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Introduction

[cite_start]Car damage assessment has traditionally been a manual, time-consuming, and inconsistent process, prone to human error[cite: 26, 68]. [cite_start]This project introduces an advanced deep learning-based system to automate car damage classification and cost estimation[cite: 27, 69]. [cite_start]By leveraging sophisticated RCNN models and data-driven methods, we offer an efficient and reliable alternative to traditional damage assessment workflows, significantly improving efficiency and customer satisfaction in insurance claim processes[cite: 25, 48, 69, 79].

## Project Objective

[cite_start]The primary objective of this project is to develop an RCNN-based intelligent computer vision solution for the automated assessment of car damage and accurate repair cost estimation[cite: 47]. [cite_start]This aims to provide an efficient, scalable approach to expedite insurance claims, eliminate human bias, and enhance customer satisfaction[cite: 48].

Key objectives include:
* [cite_start]**Automate Damage Detection:** Develop an RCNN-based model to detect whether a vehicle is damaged, replacing manual inspection with a prompt and reliable automated system[cite: 50, 51].
* [cite_start]**Classify Damage Severity:** Train the RCNN model to classify damage into `minor`, `moderate`, or `severe` levels, providing actionable insights for repair needs[cite: 52, 53].
* [cite_start]**Identify Damage Location:** Enable the model to pinpoint the specific location of damage (e.g., front bumper, rear panel, side door), enhancing real-world usability[cite: 54, 55].
* [cite_start]**Estimate Repair Costs:** Integrate a cost prediction module linked to damage severity and location for data-driven repair cost calculations, critical for expediting insurance claims[cite: 29, 56, 57].
* [cite_start]**Data Preparation and Augmentation:** Utilize web scraping to gather a comprehensive dataset of car images (damaged and undamaged), followed by annotation, preprocessing (resizing, normalization), and augmentation to improve model performance[cite: 58, 59, 74].
* [cite_start]**Comparative Model Performance:** Compare RCNN and YOLO models using metrics like accuracy, precision, and recall to select the optimal model for damage detection and classification[cite: 60, 61, 75].
* [cite_start]**Visualize Results:** Incorporate techniques to effectively display damage detection overlays, classification results, and cost estimations in a user-friendly format[cite: 62, 63].

## Key Functional Modules

The system is structured into four core components, each building upon the previous stage:

1.  [cite_start]**Car / Not Car Classification[cite: 28]:**
    * [cite_start]**Purpose:** Filters out irrelevant images that do not contain vehicles[cite: 1, 221].
    * [cite_start]**Method:** A binary image classifier distinguishes between images of cars and non-car objects[cite: 1].
    * [cite_start]**Outcome:** Only images classified as "Car" are forwarded to the next stage[cite: 1].
2.  [cite_start]**Damaged / Not Damaged Classification[cite: 28]:**
    * [cite_start]**Purpose:** Identifies whether the car in the image has visible damage[cite: 1, 230].
    * [cite_start]**Method:** A binary classifier trained on car images labeled "Damaged" or "Not Damaged," focusing on features like scratches, dents, and broken parts[cite: 1].
    * [cite_start]**Outcome:** Undamaged cars are filtered from further analysis, optimizing resource usage[cite: 1].
3.  [cite_start]**Damage Severity Classification[cite: 28]:**
    * [cite_start]**Purpose:** Categorizes the severity of the damage[cite: 1, 223].
    * [cite_start]**Categories:** Minor, Moderate, Severe[cite: 1, 29, 52].
    * [cite_start]**Method:** A multi-class classifier trained with annotated images reflecting varying degrees of damage[cite: 1].
    * [cite_start]**Significance:** Helps in repair cost estimation and insurance claim prioritization[cite: 1].
4.  [cite_start]**Damage Location Detection[cite: 28]:**
    * [cite_start]**Purpose:** Identifies the specific location of damage on the vehicle (e.g., Front, Rear, Left Side, Right Side, Roof)[cite: 1, 54, 224].
    * [cite_start]**Method:** Uses object detection techniques like R-CNN or YOLO to localize damaged areas[cite: 1].
    * [cite_start]**Output:** Bounding boxes with labeled regions (e.g., “Front Bumper - Damaged”)[cite: 1].

## Workflow Overview

The system processes images through a sequential deep learning pipeline:

[cite_start]![Data Flow Diagram](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction/blob/main/DataFlow.png) [cite: 148]

1.  [cite_start]**Image Input:** User uploads or captures an image of a car[cite: 1].
2.  [cite_start]**Preprocessing:** Images are resized, normalized, and augmented[cite: 1].
3.  **Model Pipeline Execution:**
    * [cite_start]**Module 1:** Checks if the image contains a car[cite: 1, 158, 221].
    * [cite_start]**Module 2:** Checks for damage if it is a car[cite: 1, 160, 222].
    * [cite_start]**Module 3:** Determines severity if damage is present[cite: 1, 162, 223].
    * [cite_start]**Module 4:** Detects and localizes damage[cite: 1, 164, 224].
4.  **Output:** A comprehensive report including:
    * [cite_start]Damage status [cite: 1]
    * [cite_start]Severity level [cite: 1]
    * [cite_start]Location(s) of damage with bounding boxes [cite: 1]
    * [cite_start]Estimated repair cost [cite: 29]

## Proposed System Architecture

[cite_start]Our project introduces an advanced deep learning pipeline leveraging Region-based Convolutional Neural Networks (RCNN) for automated car damage assessment and repair cost prediction[cite: 150].

### [cite_start]Data Preparation [cite: 152]
* [cite_start]**Web Scraping:** Python-based tools gathered thousands of car images from online platforms, including both damaged and undamaged vehicles, annotated with details on type, severity, and location of damage[cite: 153, 154].
* [cite_start]**Dataset Augmentation:** Techniques like flipping, rotation, cropping, and brightness adjustments were applied to simulate diverse environmental and photographic conditions, enriching dataset variability and model robustness[cite: 155, 156].

### [cite_start]RCNN Architecture [cite: 157]
* [cite_start]**Car Detection:** The initial stage of the RCNN pipeline identifies and localizes cars in images using bounding boxes, ensuring only relevant areas are processed[cite: 158, 159].
* [cite_start]**Damage Detection:** The second stage classifies detected objects as damaged or undamaged, optimizing processing efficiency[cite: 160, 161].
* [cite_start]**Severity Classification:** The third stage categorizes damage severity (minor, moderate, severe) based on features like size, shape, and intensity[cite: 162, 163].
* [cite_start]**Location Identification:** The final stage determines the specific region of the car affected by damage (e.g., front bumper, rear panel, side door), providing detailed spatial analysis for accurate cost estimation[cite: 164, 165].

### [cite_start]Cost Prediction Module [cite: 166]
* [cite_start]**Integration with RCNN:** A regression model is integrated to predict repair costs based on predefined mappings between damage severity, location, and real-world repair costs derived from the dataset[cite: 167].
* [cite_start]**Output:** The module provides a numerical estimate of the repair cost, offering actionable insights for insurance claims and repair decisions[cite: 168].

### [cite_start]Training and Validation [cite: 169]
* [cite_start]**Dataset Splitting:** The dataset was split into training, validation, and testing subsets for robust model evaluation[cite: 170].
* [cite_start]**Metrics:** Performance is evaluated using Accuracy, Precision, Recall, and F1-Score[cite: 171, 172, 174, 176, 178].

## Technologies and Tools Used

* [cite_start]**Programming Language:** Python [cite: 1]
* [cite_start]**Deep Learning Frameworks:** TensorFlow / Keras [cite: 1]
* [cite_start]**Models:** Convolutional Neural Networks (CNNs), Region-based CNN (R-CNN), YOLO [cite: 1]
* [cite_start]**Image Processing:** OpenCV (for pre-processing tasks like resizing, augmentation) [cite: 1]
* [cite_start]**Hardware:** GPU-enabled system (recommended for faster training and inference) [cite: 1]

## Dataset Description

[cite_start]For this project, we created a vast and varied dataset by combining web scraping with publicly available datasets, primarily from Kaggle[cite: 131, 136].

* [cite_start]**Data Collection:** Approximately 200 images were collected using Python-based web scraping tools from various online sources, including car repair websites and insurance claim portals[cite: 133, 134, 153].
* [cite_start]**Diversity:** The dataset includes a wide range of car models, types of damage, and different environmental conditions (lighting, angles, severity) to reflect real-world variations[cite: 134, 135, 143].
* [cite_start]**Annotations:** Images feature detailed annotations indicating whether the car is damaged/undamaged, the severity of damage (minor, moderate, severe), and its specific location (front bumper, rear panel, etc.)[cite: 137, 144, 154].
* [cite_start]**Augmentation:** Data augmentation techniques like rotation, flipping, cropping, and brightness adjustments were applied to boost the model's generalization capabilities and simulate diverse photographic conditions[cite: 145, 155, 156].
* **Dataset Splitting:** The dataset was divided into:
    * [cite_start]**Training Dataset:** A larger portion for RCNN model pattern learning[cite: 140].
    * [cite_start]**Validation Dataset:** A smaller subset used to evaluate performance during training and fine-tune hyperparameters[cite: 141].

## Employed Models

[cite_start]Both RCNN and YOLO models were employed and compared to achieve a balance of accuracy and efficiency for diverse use-case scenarios[cite: 207, 210].

### [cite_start]RCNN (Region-based Convolutional Neural Network) [cite: 184]
* **Mechanism:** Divides an image into regions of interest (RoIs) which are then processed individually for classification and bounding box prediction. [cite_start]This ensures higher accuracy at the cost of speed[cite: 185, 186].
* **Our Use:**
    * [cite_start]**Car Detection:** Trained to detect cars and extract bounding boxes for further processing[cite: 188, 189].
    * [cite_start]**Damage Detection:** Identified damage, classifying severity (minor, moderate, severe) by analyzing patterns like dents, cracks, and scratches[cite: 190, 191].
    * [cite_start]**Location Identification:** Pinpointed specific damage locations (e.g., front bumper, side door, rear panel)[cite: 192, 193].
* [cite_start]**Characteristics:** High precision for detailed damage analysis but requires more computation time[cite: 194, 249].

### [cite_start]YOLO (You Only Look Once) [cite: 195]
* [cite_start]**Mechanism:** Processes the entire image in a single step, dividing it into a grid and predicting bounding boxes and class probabilities simultaneously, prioritizing speed without significant accuracy compromise[cite: 196, 197].
* **Our Use:**
    * [cite_start]**Car Detection:** Rapidly detected cars in dataset images, providing real-time bounding boxes[cite: 199, 200].
    * [cite_start]**Damage Detection and Classification:** Identified and classified damages into severity levels in one go[cite: 201, 202].
    * [cite_start]**Location Identification:** Effectively located damaged areas (rear, front, sides) by assigning grid cells to damaged zones for quick analysis[cite: 203, 204].
* [cite_start]**Characteristics:** Real-time capabilities, suitable for scenarios requiring immediate assessment[cite: 205, 252].

### [cite_start]Model Comparison [cite: 206]

| Aspect                      | RCNN                                    | YOLO                                            |
| :-------------------------- | :-------------------------------------- | :---------------------------------------------- |
| **Speed** | Slower due to region-based analysis     | Faster due to single-pass processing            |
| **Accuracy** | Higher for detailed classifications     | Slightly lower but competitive                  |
| **Use Case** | Suitable for detailed offline analysis  | Ideal for real-time assessments                 |
| **Damage Severity Analysis**| More precise for subtle damage patterns | Faster but less nuanced for complex damage      |
| **Cost Prediction** | Provides detailed data for cost estimation | Provides quick overviews for cost inputs         |
| **Accuracy Score (%)** | [cite_start]60.08 [cite: 206, 244]                  | [cite_start]43 [cite: 206, 244]                             |

## Novelty

[cite_start]The innovative aspect of this project lies in its **Cost Prediction Module**[cite: 212]. [cite_start]Unlike many existing systems that only classify damage into broad categories, our system quantifies severity as a percentage, providing a more precise estimation directly related to the extent of the damage[cite: 213]. [cite_start]Furthermore, it integrates the damage's location (front, rear, sides, undercarriage) as a key factor, as different areas require varying complexities and resources for repairs[cite: 214].

By combining these two elements, the system offers personalized and accurate cost predictions tailored to each vehicle's specific damage scenario. [cite_start]This advancement not only enhances car damage classification but also delivers valuable insights for insurance companies, repair shops, and vehicle owners, improving decision-making processes and overall service quality[cite: 215, 216]. [cite_start]No previous research has successfully integrated damage detection, severity classification, location identification, and cost prediction into a single framework[cite: 128].

## Results and Discussion

[cite_start]The performance of the Car Damage Classification and Cost Prediction System, consisting of four modules, was primarily evaluated using the RCNN model[cite: 218].

* [cite_start]**Car Detection (Module 1):** Achieved an accuracy of **92%**[cite: 229]. [cite_start]The RCNN model performed well in detecting cars across various settings, despite challenges like varying lighting, background complexity, and different car models[cite: 228, 229].
* [cite_start]**Damage Detection (Module 2):** Showed good accuracy of **86%**[cite: 233]. [cite_start]It effectively identified damaged vs. undamaged cars, though some false positives occurred due to environmental factors (e.g., reflective surfaces, minor scratches)[cite: 232, 233].
* [cite_start]**Damage Severity Classification (Module 3):** Achieved **82%** precision for class 0 (likely undamaged/minor), **41%** for class 1 (moderate), and **58%** for class 2 (severe)[cite: 237]. [cite_start]It performed well in distinguishing severe damage but struggled with differentiating between minor and moderate damage, especially for subtle damages[cite: 236].
* [cite_start]**Damage Location Identification (Module 4):** Had the lowest accuracy, facing challenges due to complex shapes and overlapping damage areas[cite: 240, 241]. [cite_start]Its training accuracy was **60.08%**[cite: 206, 262].

### [cite_start]Model Comparison Results [cite: 242]
* [cite_start]**RCNN Performance:** Demonstrated superior training accuracy of **60.08%**[cite: 244]. [cite_start]It excelled in detecting subtle damages and classifying severity levels, providing detailed spatial insights into damage locations, making it highly reliable for detailed assessments in insurance and repair cost prediction[cite: 248, 249].
* [cite_start]**YOLO Performance:** Achieved a training accuracy of **43%**[cite: 244]. [cite_start]While faster and efficient for real-time applications, its overall accuracy was lower due to its grid-based approach, which sometimes missed finer damage details[cite: 251, 252].

[cite_start]The results highlight RCNN’s effectiveness in achieving higher accuracy for detailed damage analysis and cost prediction, while YOLO offers unparalleled speed for rapid preliminary assessments[cite: 253, 254]. [cite_start]This trade-off guides model selection based on specific use-case requirements: RCNN for thorough analysis (insurance reports, repair estimates) and YOLO for rapid assessments (initial scans, customer-facing applications)[cite: 256, 257, 258].

## Conclusion

[cite_start]The Car Damage Classification and Cost Prediction System successfully leverages advanced RCNN techniques to identify car damages and estimate repair costs[cite: 260]. [cite_start]The system's four main modules (car detection, damage detection, severity classification, and damage location identification) demonstrate the promising role of deep learning in automotive damage assessment[cite: 261, 265]. [cite_start]While car detection and damage detection achieved impressive accuracies (85.47% and 83.20% respectively) [cite: 262][cite_start], the damage location identification module presented challenges, reaching 60.08% accuracy due to complex and overlapping damage areas[cite: 262]. [cite_start]RCNN proved superior to YOLO for detailed object localization and severity estimation[cite: 263]. [cite_start]The integrated cost prediction feature, based on damage severity and location, provides valuable insights for real-time assessments in insurance and repair contexts[cite: 264]. [cite_start]Further enhancements in image resolution and training data could improve accuracy, especially in more complex tasks[cite: 265].

## Installation & Setup

To get a copy of the project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    [cite_start]git clone [https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction.git](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction.git) [cite: 14]
    cd Car_damage_classification-Cost_prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    *(You will need a `requirements.txt` file in your repository listing all necessary libraries)*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download pre-trained models/weights:**
    *(Instructions on where to download the `.h5` model files (e.g., `car_bike_classification_model.h5`, `car_damage_model.h5`, `car_damage_location_model.h5`, `car_damage_severity_model.h5`) and where to place them in the project structure, typically in a `models/` directory)*

## Usage

This project includes a Flask application for demonstrating the car damage detection and cost prediction.

1.  **Ensure all dependencies are installed and models are placed correctly.**
2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
3.  **Access the web interface:** Open your web browser and navigate to `http://127.0.0.1:5000/`.
4.  **Upload an image:** Use the "Choose File" and "Upload and Analyze" buttons to process a car image. The results will display:
    * Whether the object is a car.
    * Whether the car is damaged.
    * Damage Confidence.
    * Damage Locations with confidence.
    * Damage Severity Label and Confidence.
    * Estimated Repair Cost.

    ![Car Damage Detection Screenshot](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction/blob/main/output_screenshot.png)

## Applications

The developed system has diverse real-world applications:
* [cite_start]**Insurance Industry:** Automated claim processing, fraud detection, and objective damage assessment[cite: 1, 25, 42, 85].
* [cite_start]**Car Rental Services:** Efficient pre- and post-rental vehicle condition assessments, improving dispute resolution[cite: 1].
* [cite_start]**Automobile Workshops:** Streamlined damage diagnostics for repair estimation, enhancing customer service[cite: 1].
* [cite_start]**Fleet Management:** Facilitates continuous monitoring and proactive maintenance planning for vehicle fleets[cite: 1].

## Future Enhancements

We plan to continuously improve this project with the following enhancements:
* [cite_start]**Mobile App Integration:** Develop a mobile application for real-time damage detection[cite: 1].
* [cite_start]**3D Reconstruction:** Incorporate 3D vision techniques to assess internal or less visible damage[cite: 1].
* [cite_start]**Synthetic Data Generation:** Utilize GANs (Generative Adversarial Networks) to create synthetic damaged car images, enhancing model robustness and accuracy[cite: 1].
* [cite_start]**Repair Cost Estimation Model:** Further refine the module to provide more precise repair cost estimations based on detected damage[cite: 1].

## Contributing

We welcome contributions to this project! If you'd like to contribute, please fork the repository, create a new branch, make your changes, and submit a pull request. Ensure your code adheres to good coding practices and includes relevant tests.


## Contact

For any questions or collaborations, feel free to reach out to the project team:

* **Arun S:** [Your GitHub Profile Link or Email]
* [cite_start]**Laxsman Karthik S:** [https://github.com/laxsman-karthik-s](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction) [cite: 14]


<img width="701" height="150" alt="image" src="https://github.com/user-attachments/assets/a15c1af0-9d4f-4fef-bc62-77919dbaea06" />
<img width="676" height="698" alt="image" src="https://github.com/user-attachments/assets/f4622b68-cda5-452f-8e17-d911620a6b8f" />

---
