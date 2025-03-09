# NSL-KDD Intrusion Detection Project

This repository provides a machine learning pipeline for detecting network intrusions using the NSL-KDD dataset. The project includes data preprocessing, model training, and a RESTful Flask API for making predictions.

---

## **Project Structure**

```plaintext
IDS-ML-Project/
│
├── src/
│   ├── preprocess.py    # Module for loading and preprocessing NSL-KDD data
│   ├── train.py         # Script to train the Random Forest model with SMOTE balancing
│   ├── app.py           # Flask API to serve model predictions
│   ├──evaluate.py       # Script for model evaluation (confusion matrix, ROC, etc.)
|
├── data/                # Folder containing the NSL-KDD dataset (KDDTrain+.txt, KDDTest+.txt)
│   ├── KDDTrain+.txt       # Training dataset (NSL-KDD)
│   └── KDDTest+.txt        # Testing dataset (NSL-KDD)
├── models/
│   └── random_forest.joblib  # Saved Random Forest model
│
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
```

---

## **Features**

- **Data Preprocessing**:  
  - Rename columns for clarity (e.g., converting column 41 to `label` and column 42 to `difficulty`).
  - Encode categorical values (`protocol_type`, `service`, and `flag`) into numeric values using `LabelEncoder`.
  - Split data into features (`X`) and labels (`y`) for training and testing.

- **Model Training and Evaluation**:  
  - Train a Random Forest classifier on the preprocessed data.  
  - Evaluate the model’s performance using metrics like accuracy, confusion matrix, and classification report.

- **Flask API for Predictions**:  
  - Serve the trained model via a RESTful API.  
  - Provide an endpoint (`/predict`) that accepts JSON input and returns predictions (e.g., "normal", "neptune", etc.).

---

## **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bilal-66/IDS-ML-Project.git
   cd IDS-ML-Project
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How to Run**

### **1. Data Preprocessing and Training**

1. Ensure the **NSL-KDD dataset** (`KDDTrain+.txt` and `KDDTest+.txt`) is placed inside the `data/` folder.  
2. Run the training script:
   ```bash
   python src/train.py
   ```
   - This will preprocess the data, train the Random Forest model, evaluate its performance, and save the trained model in the `models/` directory.

---

### **2. Serving Predictions**

1. Start the Flask API:
   ```bash
   python src/app.py
   ```
2. The API will be available at:
   ```
   http://127.0.0.1:5000
   ```

---

### **3. Making Predictions**

You can send a POST request to the `/predict` endpoint. For example, using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"features":[0.1, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2, 13.3, 14.4, 15.5, 16.6, 17.7, 18.8, 19.9, 20.0, 21.1, 22.2, 23.3, 24.4, 25.5, 26.6, 27.7, 28.8, 29.9, 30.0, 31.1, 32.2, 33.3, 34.4, 35.5, 36.6, 37.7, 38.8, 39.9, 40.0]}' \
http://127.0.0.1:5000/predict
```

**Example Response:**
```json
{"prediction": "normal"}
```

---

## **Dataset**

The NSL-KDD dataset is a widely used benchmark dataset for evaluating intrusion detection systems. It contains both normal and attack traffic, with a variety of attack types such as DoS, R2L, U2R, and Probe.

For more information about the dataset, visit the official NSL-KDD page:  
[NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)

---

## **Contributing**

Feel free to open issues or pull requests if you find bugs, have suggestions, or want to contribute new features.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

