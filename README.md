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

Car damage assessment has traditionally been a manual, time-consuming, and inconsistent process, prone to human error. This project introduces an advanced deep learning-based system to automate car damage classification and cost estimation. By leveraging sophisticated RCNN models and data-driven methods, we offer an efficient and reliable alternative to traditional damage assessment workflows, significantly improving efficiency and customer satisfaction in insurance claim processes.

## Project Objective

The primary objective of this project is to develop an RCNN-based intelligent computer vision solution for the automated assessment of car damage and accurate repair cost estimation. This aims to provide an efficient, scalable approach to expedite insurance claims, eliminate human bias, and enhance customer satisfaction.

Key objectives include:
* **Automate Damage Detection:** Develop an RCNN-based model to detect whether a vehicle is damaged, replacing manual inspection with a prompt and reliable automated system.
* **Classify Damage Severity:** Train the RCNN model to classify damage into `minor`, `moderate`, or `severe` levels, providing actionable insights for repair needs.
* **Identify Damage Location:** Enable the model to pinpoint the specific location of damage (e.g., front bumper, rear panel, side door), enhancing real-world usability.
* **Estimate Repair Costs:** Integrate a cost prediction module linked to damage severity and location for data-driven repair cost calculations, critical for expediting insurance claims.
* **Data Preparation and Augmentation:** Utilize web scraping to gather a comprehensive dataset of car images (damaged and undamaged), followed by annotation, preprocessing (resizing, normalization), and augmentation to improve model performance.
* **Comparative Model Performance:** Compare RCNN and YOLO models using metrics like accuracy, precision, and recall to select the optimal model for damage detection and classification.
* **Visualize Results:** Incorporate techniques to effectively display damage detection overlays, classification results, and cost estimations in a user-friendly format.

## Key Functional Modules

The system is structured into four core components, each building upon the previous stage:

1.  **Car / Not Car Classification:**
    * **Purpose:** Filters out irrelevant images that do not contain vehicles.
    * **Method:** A binary image classifier distinguishes between images of cars and non-car objects.
    * **Outcome:** Only images classified as "Car" are forwarded to the next stage.
2.  **Damaged / Not Damaged Classification:**
    * **Purpose:** Identifies whether the car in the image has visible damage.
    * **Method:** A binary classifier trained on car images labeled "Damaged" or "Not Damaged," focusing on features like scratches, dents, and broken parts.
    * **Outcome:** Undamaged cars are filtered from further analysis, optimizing resource usage.
3.  **Damage Severity Classification:**
    * **Purpose:** Categorizes the severity of the damage.
    * **Categories:** Minor, Moderate, Severe.
    * **Method:** A multi-class classifier trained with annotated images reflecting varying degrees of damage.
    * **Significance:** Helps in repair cost estimation and insurance claim prioritization.
4.  **Damage Location Detection:**
    * **Purpose:** Identifies the specific location of damage on the vehicle (e.g., Front, Rear, Left Side, Right Side, Roof).
    * **Method:** Uses object detection techniques like R-CNN or YOLO to localize damaged areas.
    * **Output:** Bounding boxes with labeled regions (e.g., “Front Bumper - Damaged”).

## Workflow Overview

The system processes images through a sequential deep learning pipeline:

![Data Flow Diagram](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction/blob/main/DataFlow.png)

1.  **Image Input:** User uploads or captures an image of a car.
2.  **Preprocessing:** Images are resized, normalized, and augmented.
3.  **Model Pipeline Execution:**
    * **Module 1:** Checks if the image contains a car.
    * **Module 2:** Checks for damage if it is a car.
    * **Module 3:** Determines severity if damage is present.
    * **Module 4:** Detects and localizes damage.
4.  **Output:** A comprehensive report including:
    * Damage status
    * Severity level
    * Location(s) of damage with bounding boxes
    * Estimated repair cost

## Proposed System Architecture

Our project introduces an advanced deep learning pipeline leveraging Region-based Convolutional Neural Networks (RCNN) for automated car damage assessment and repair cost prediction.

### Data Preparation
* **Web Scraping:** Python-based tools gathered thousands of car images from online platforms such as car repair websites and insurance claim portals. The dataset includes images of both damaged and undamaged cars, annotated with details about damage type, severity, and location.
* **Dataset Augmentation:** To ensure model robustness, data augmentation techniques such as flipping, rotation, cropping, and brightness adjustments were applied. These techniques simulate diverse environmental and photographic conditions, enriching the dataset's variability.

### RCNN Architecture
* **Car Detection:** The first stage of the RCNN pipeline identifies and localizes cars in the images using bounding boxes. This ensures that only relevant areas of the image are processed in subsequent steps.
* **Damage Detection:** In the second stage, the model classifies the detected objects as damaged or undamaged. This binary classification filters out undamaged cars, optimizing processing efficiency.
* **Severity Classification:** The third stage focuses on categorizing the severity of damage into levels such as minor, moderate, or severe. This model leverages features like the size, shape, and intensity of the damage to deliver precise predictions.
* **Location Identification:** The final stage determines the specific region of the car affected by damage (e.g., front bumper, rear panel, side door). This spatial analysis provides a detailed overview of the damage, facilitating accurate cost estimation.

### Cost Prediction Module
* **Integration with RCNN:** A regression model is integrated with the RCNN pipeline to predict repair costs based on predefined mappings between damage severity, location, and real-world repair costs derived from the dataset.
* **Output:** The module provides a numerical estimate of the repair cost, offering actionable insights for insurance claims and repair decisions.

### Training and Validation
* **Dataset Splitting:** The dataset was split into training, validation, and testing subsets to ensure robust evaluation of model performance.
* **Metrics:** Performance is evaluated using Accuracy, Precision, Recall, and F1-Score.

## Technologies and Tools Used

* **Programming Language:** Python
* **Deep Learning Frameworks:** TensorFlow / Keras
* **Models:** Convolutional Neural Networks (CNNs), Region-based CNN (R-CNN), YOLO
* **Image Processing:** OpenCV (for pre-processing tasks like resizing, augmentation)
* **Hardware:** GPU-enabled system (recommended for faster training and inference)

## Dataset Description

For this project, we created a vast and varied dataset by combining web scraping with publicly available datasets, primarily from Kaggle.

* **Data Collection:** Approximately 200 images were collected using Python-based web scraping tools from various online sources, including car repair websites and insurance claim portals.
* **Diversity:** The dataset includes a wide range of car models, types of damage, and different environmental conditions (lighting, angles, severity) to reflect real-world variations.
* **Annotations:** Images feature detailed annotations indicating whether the car is damaged/undamaged, the severity of damage (minor, moderate, severe), and its specific location (front bumper, rear panel, etc.).
* **Augmentation:** Data augmentation techniques like rotation, flipping, cropping, and brightness adjustments were applied to boost the model's generalization capabilities and simulate diverse photographic conditions.
* **Dataset Splitting:** The dataset was divided into:
    * **Training Dataset:** A larger portion for RCNN model pattern learning.
    * **Validation Dataset:** A smaller subset used to evaluate performance during training and fine-tune hyperparameters.

## Employed Models

Both RCNN and YOLO models were employed and compared to achieve a balance of accuracy and efficiency for diverse use-case scenarios.

### RCNN (Region-based Convolutional Neural Network)
* **Mechanism:** RCNN operates by dividing an image into regions of interest (RoIs), which are then processed individually for classification and bounding box prediction. This region-wise processing ensures higher accuracy but comes at the cost of speed.
* **Our Use:**
    * **Car Detection:** RCNN was trained to detect cars in images, extracting bounding boxes around vehicles for further processing.
    * **Damage Detection:** The algorithm identified whether a detected car exhibited damage and classified the severity (minor, moderate, severe) by analyzing patterns such as dents, cracks, and surface scratches.
    * **Location Identification:** RCNN pinpointed specific damage locations (e.g., front bumper, side door, rear panel), providing precise spatial information for cost estimation.
* **Characteristics:** This step-by-step approach made RCNN a valuable tool for detailed damage analysis, offering high precision but requiring more computation time.

### YOLO (You Only Look Once)
* **Mechanism:** YOLO processes the entire image in a single step, dividing it into a grid and predicting bounding boxes and class probabilities simultaneously. This approach emphasizes speed without significantly compromising accuracy.
* **Our Use:**
    * **Car Detection:** YOLO rapidly detected cars in the dataset images, providing bounding boxes in real time.
    * **Damage Detection and Classification:** YOLO was used to identify and classify damages into severity levels in one go, bypassing the region-wise processing of RCNN.
    * **Location Identification:** YOLO effectively located damaged areas on vehicles (e.g., rear, front, sides) by assigning grid cells to damaged zones, facilitating quick and efficient analysis.
* **Characteristics:** YOLO’s real-time capabilities made it particularly suitable for scenarios requiring immediate assessment, such as in insurance claim systems.

### Model Comparison

| Aspect                      | RCNN                                    | YOLO                                            |
| :-------------------------- | :-------------------------------------- | :---------------------------------------------- |
| **Speed** | Slower due to region-based analysis     | Faster due to single-pass processing            |
| **Accuracy** | Higher for detailed classifications     | Slightly lower but competitive                  |
| **Use Case** | Suitable for detailed offline analysis  | Ideal for real-time assessments                 |
| **Damage Severity Analysis**| More precise for subtle damage patterns | Faster but less nuanced for complex damage      |
| **Cost Prediction** | Provides detailed data for cost estimation | Provides quick overviews for cost inputs         |
| **Accuracy Score (%)** | 60.08                  | 43                             |

## Novelty

The innovative aspect of this project is its Cost Prediction Module, which calculates repair costs by considering two key factors: damage severity percentage and damage location. Unlike many existing systems that only classify damage into broad categories, this project quantifies severity as a percentage, providing a more precise estimation that directly relates to the extent of the damage. Additionally, the location of the damage—whether on the front, rear, sides, or undercarriage—significantly influences repair costs, as different areas require varying levels of complexity and resources for repairs. By integrating these two elements, the system offers personalized and accurate cost predictions tailored to each vehicle's specific damage scenario. This advancement not only enhances the effectiveness of car damage classification but also delivers valuable insights for insurance companies, repair shops, and vehicle owners, improving their decision-making processes and overall service quality.

## Results and Discussion

The performance of the Car Damage Classification and Cost Prediction System, which consists of four modules, was evaluated using the RCNN (Region Convolutional Neural Network) model.

* **Car Detection (Module 1):** The RCNN model performed well in detecting whether the object in the image was a car or not, accurately identifying cars in a variety of settings, despite challenges such as varying lighting conditions, background complexity, and different car models. It achieved an accuracy of **92%**.
* **Damage Detection (Module 2):** The damage detection module performed with good accuracy of **86%**, identifying whether a car was damaged or undamaged. However, some false positives were detected where undamaged cars were incorrectly classified as damaged, likely due to environmental factors (e.g., reflective surfaces or minor scratches).
* **Damage Severity Classification (Module 3):** The severity classification module struggled in some cases to distinguish between minor and moderate damage, particularly for less obvious or smaller damages. It performed well in distinguishing severe damage, as the model was able to identify significant damage areas with higher confidence.
* **Damage Location Identification (Module 4):** The location identification module, which uses RCNN to detect specific regions of the car (e.g., front, side, or rear), had the lowest accuracy among the four modules, reaching **60.08%**. This may be due to complex shapes or overlapping areas of damage, which made it difficult for the model to distinguish exact locations accurately.

<img width="701" height="150" alt="image" src="https://github.com/user-attachments/assets/b7cc4ec8-11be-4ba3-8cad-417f6491957d" />
<img width="676" height="698" alt="image" src="https://github.com/user-attachments/assets/4d68f74a-c37f-4c2c-bf10-c58545fd819e" />


### Model Comparison Results

A comparison of model performance metrics reveals the effectiveness of RCNN and YOLO in the Car Damage Classification and Cost Prediction system. With a training accuracy of 60.08%, RCNN outperformed YOLO, which achieved a training accuracy of 43%. Both models were assessed on their ability to detect car damage, classify its severity, and identify damage locations, with RCNN consistently providing better results.

* **RCNN Performance:** RCNN demonstrated superior accuracy compared to YOLO, especially in detecting subtle damages and correctly classifying severity levels. The region-based analysis enabled RCNN to provide detailed spatial insights into damage locations, making it highly reliable for detailed assessments required in insurance and repair cost prediction.
* **YOLO Performance:** While YOLO's speed and efficiency made it suitable for real-time applications, its overall accuracy was lower due to its grid-based approach, which occasionally missed finer details in damage patterns. Despite this, YOLO excelled in providing rapid assessments, which can be useful in scenarios demanding quick decisions.

The results highlight RCNN’s effectiveness in achieving higher accuracy, making it the preferred model for detailed damage analysis and cost prediction. YOLO, while not as precise, offers unparalleled speed and efficiency, making it valuable for real-time applications such as preliminary assessments or field-based evaluations. The stark contrast in accuracy underscores the trade-off between precision and speed in model selection: RCNN is ideal for use cases demanding thorough analysis and accuracy, such as generating insurance reports or repair estimates, while YOLO is suitable for scenarios where rapid assessments are prioritized, such as initial damage scans or customer-facing applications. These findings offer stakeholders clear insights into the applicability of each model and guide their selection based on specific use-case requirements.

## Conclusion

The Car Damage Classification and Cost Prediction System developed in this project leverages advanced RCNN (Region Convolutional Neural Network) techniques to effectively identify car damages and estimate repair costs. The system consists of four main modules: car detection, damage detection, severity classification, and damage location identification. While the car and damage detection modules achieved impressive accuracies of 92% and 86%, respectively, the damage location identification module faced challenges, reaching only 60.08% due to overlapping damage areas and complex car shapes. RCNN proved to be superior to other models like YOLO, particularly for detailed object localization and severity estimation. Additionally, the integrated cost prediction feature, based on damage severity and location, provides valuable insights for real-time assessments in insurance and repair contexts. Overall, this project highlights the promising role of deep learning and computer vision in car damage assessment, though further enhancements in image resolution and training data could improve accuracy, especially in more complex tasks.

## Installation & Setup

To get a copy of the project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction.git](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction.git)
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
* **Insurance Industry:** Automated claim processing, fraud detection, and objective damage assessment.
* **Car Rental Services:** Efficient pre- and post-rental vehicle condition assessments, improving dispute resolution.
* **Automobile Workshops:** Streamlined damage diagnostics for repair estimation, enhancing customer service.
* **Fleet Management:** Facilitates continuous monitoring and proactive maintenance planning for vehicle fleets.

## Future Enhancements

We plan to continuously improve this project with the following enhancements:
* **Mobile App Integration:** Develop a mobile application for real-time damage detection.
* **3D Reconstruction:** Incorporate 3D vision techniques to assess internal or less visible damage.
* **Synthetic Data Generation:** Utilize GANs (Generative Adversarial Networks) to create synthetic damaged car images, enhancing model robustness and accuracy.
* **Repair Cost Estimation Model:** Further refine the module to provide more precise repair cost estimations based on detected damage.

## Contributing

We welcome contributions to this project! If you'd like to contribute, please fork the repository, create a new branch, make your changes, and submit a pull request. Please ensure your code adheres to good coding practices and includes relevant tests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
*(Create a `LICENSE` file in your repository with the MIT License text if you haven't already).*

## Contact

For any questions or collaborations, feel free to reach out to the project team:

* **Arun S:** [Your GitHub Profile Link or Email]
* **Laxsman Karthik S:** [https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction](https://github.com/laxsman-karthik-s/Car_damage_classification-Cost_prediction)

---
