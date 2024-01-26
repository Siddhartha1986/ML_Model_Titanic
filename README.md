
# ML Titanic Random Forest Model

## Overview
This project presents a machine learning model that predicts the probability of a passenger's death on the Titanic. The model is based on two primary features: gender and ticket class.

## Features and Target
- **Features:** Gender, Ticket Class
- **Target:** Death Rate

## Dataset
The Titanic dataset was carefully analyzed and prepared to train the machine learning model. This dataset provides detailed information about the passengers, which is crucial for predicting survival outcomes.

## Machine Learning Model
- The model utilizes the **K Nrearest Neighbours** algorithm.
- It achieved an accuracy of **79.85%** in predicting the survival outcomes.

## User Interface
- A User Interface (UI) was developed using **Streamlit**. 
- This UI allows users to input passenger details and receive survival predictions interactively.

## Deployment
- The model, along with its UI, is deployed on the **Streamlit.io** platform.
- Users can access and interact with the model via the deployed application.

## Project Structure
- `app.py`: The main Streamlit application script.
- `new_model.py`: Script for the Random Forest model.
- `prediction.py`: Handles the prediction logic.
- `data`: Directory containing the dataset used.
- `output_models`: Contains the trained model files.
- `requirements.txt`: Lists all the dependencies for the project.

## Installation and Usage
- Clone this repository to your local machine.
- Install the necessary dependencies: `pip install -r requirements.txt`
- Run the Streamlit application: `streamlit run app.py`

## License
This project is licensed under the [MIT Licence](LICENSE) file in this repository.

#[The link to UI for the model](https://siddhartha1986-ml-model-titanic-app-88wbxp.streamlit.app/)



