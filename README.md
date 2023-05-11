# Covid-19 Detection with Tuned CNN and TensorFlow Lite

Predicting the presence of COVID-19 from CAT Scans. Completed as a part of NYU's Advanced Machine Learning Course.

This repository is dedicated to the development and implementation of a sophisticated Covid-19 detection system using a Optimized Convolutional Neural Network (CNN) model. A version for edge devices have been created with the use of the TensorFlow Lite model. 

The primary focus of this project is to create a portable, efficient, and highly accurate model that can identify the presence of Covid-19 through image analysis, specifically lung CAT scans. This project leverages the ResNet50 and VGG16 frameworks, augmenting them with deeper layers trained specifically on an applicable dataset. Additional hyperparameter tuning optimizes the model to performance with an accuracy score of near 95%.

With the help of the TensorFlow Lite library, we then convert this model into a compact, lightweight format, suitable for use on mobile and edge devices. This facilitates real-time analysis and prediction, paving the way for rapid detection and potentially life-saving interventions on mobile devices and low powered computers as an aid for health professionals. 

**Features**

* Utilization of Convolutional Neural Networks (CNNs) to maximize efficiency and accuracy.
* Deployment on edge devices using TensorFlow Lite for real-time prediction.
* Comprehensive training and testing datasets for robust model validation.
* Easy-to-follow codebase, thoroughly documented for both beginners and advanced users.
* Detailed performance metrics and visualization tools to understand the model's effectiveness.
* Please note that while our aim is to provide a powerful tool to aid in the detection of Covid-19, it is not intended to replace professional medical advice or diagnosis. Always consult a healthcare professional for medical concerns.

## Running the Program in Google Colab

To run the COVID-19 detection Program in Google Colab, follow the steps outlined below:

1. Open the analysis.ipynb file in Google Colab. 
2. Download the data for the project and upload it either to your runtime or to your Google Drive. You can find the Dataset at this [Kaggle Competition](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).
3. Set the path variable to the location of where your data is stored within Google Drive/the session. This can be done by modifying the `path` variable in the notebook. Replace `'path/to/data'` with the actual path to the dataset on your Google Drive.

    ```python
    path = 'path/to/data'
    ```

4. Mount your Google Drive by executing the code cell that prompts you to authenticate and authorize access to your drive. This is required to be able to access the data stored in your runtime. 

5. Run the remaining code cells in the notebook sequentially. The notebook is well documented with a concise table of contents specify exactly what section or where a specific portion of the code will be run. Please feel free to raise an issue if you are having trouble!


## Contributions

This is an open-source project. We welcome contributions, ideas, and suggestions to improve the model's effectiveness and usability. Please follow these simple rules for making contributions effectively. 

* **For the Repository** - please create your own version of the Repository and make your changes on your own version. 
* **Commit and Push** - once you have made the necessary changes, please push to `master` and submit a PR. 
* **Submit a PR**: Please open a PR to merge into the original repository. 
  * You will need the approval of a maintainer to be able to push to `master`

Thank you for considering contributing to this project. Your efforts can make a significant difference in advancing COVID-19 detection capabilities. Let's work together to create a better, more robust solution!

Stay Safe, Stay Healthy.
