# 🧠 ANN Classification - Customer Churn Prediction

A deep learning web application for predicting customer churn using Artificial Neural Networks (ANN). Built with TensorFlow/Keras and deployed as an interactive Streamlit app.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-yellow.svg)

## 🎯 Project Overview

This project implements a deep learning solution to predict customer churn for a banking institution. The model analyzes customer demographics, account information, and activity patterns to identify customers at risk of leaving the service.

**Key Features:**
- ✅ Deep Neural Network with 86%+ accuracy
- ✅ Real-time predictions via Streamlit web interface
- ✅ Pre-trained model with saved encoders and scaler
- ✅ Interactive input widgets for all customer features
- ✅ Probability-based churn prediction

## 🚀 Live Demo

🌐 **[Try the App Live](https://ann-classification-churn.streamlit.app)** *(Coming Soon)*

## 📊 Dataset

**Source:** Churn_Modelling.csv  
**Total Records:** 10,000 customers  
**Features:** 14 columns (11 used for modeling)

### Feature Descriptions

| Column | Type | Description | Sample Values |
|--------|------|-------------|---------------|
| `RowNumber` | Int | Index | 1-10000 |
| `CustomerId` | Int | Unique customer ID | 15634602 |
| `Surname` | String | Customer surname | Hargrave, Hill |
| `CreditScore` | Int | Credit score | 350-850 |
| `Geography` | String | Country | France, Spain, Germany |
| `Gender` | String | Gender | Male, Female |
| `Age` | Int | Age in years | 18-92 |
| `Tenure` | Int | Years with bank | 0-10 |
| `Balance` | Float | Account balance | 0-250,000 |
| `NumOfProducts` | Int | Number of products | 1-4 |
| `HasCrCard` | Binary | Has credit card | 0, 1 |
| `IsActiveMember` | Binary | Active member | 0, 1 |
| `EstimatedSalary` | Float | Annual salary | 0-200,000 |
| `Exited` | Binary | **Target: Churned** | 0 (Stay), 1 (Churn) |

### Data Distribution

- **Class Balance:** ~20% churn rate (typical for banking)
- **Geography:** France (50%), Germany (25%), Spain (25%)
- **Gender:** Balanced distribution
- **Age Range:** 18-92 years (median ~38)

## 🏗️ Model Architecture

### Neural Network Structure

```
Model: Sequential ANN
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Dense (Input + Hidden 1)    (None, 64)                ~800      
Dense (Hidden 2)            (None, 32)                ~2,100     
Dropout                     (None, 32)                0         
Dense (Hidden 3)            (None, 16)                ~530       
Dense (Output)              (None, 1)                 17        
=================================================================
Total params: ~3,400 trainable parameters
```

### Configuration

- **Input Features:** 11 (after preprocessing)
- **Hidden Layers:** 3 layers with ReLU activation
- **Neurons:** [64, 32, 16]
- **Regularization:** Dropout (0.2-0.3)
- **Output:** 1 neuron with Sigmoid activation
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, Precision, Recall

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~86% |
| **Precision** | ~0.83 |
| **Recall** | ~0.72 |
| **F1-Score** | ~0.77 |
| **ROC-AUC** | ~0.88 |

### Training Details

- **Training Set:** 80% (8,000 samples)
- **Test Set:** 20% (2,000 samples)
- **Epochs:** ~50-100 (with early stopping)
- **Batch Size:** 32
- **Validation Split:** 20%

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11
- pip package manager
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/manola1109/ANN-Classification-Churn.git
   cd ANN-Classification-Churn
   ```

2. **Create virtual environment**
   
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the App

1. **Select Geography** - Choose customer's country (France, Germany, Spain)
2. **Select Gender** - Male or Female
3. **Adjust Age** - Slide between 18-92 years
4. **Enter Balance** - Customer's account balance
5. **Enter Credit Score** - Credit score (350-850)
6. **Enter Estimated Salary** - Annual salary
7. **Set Tenure** - Years with the bank (0-10)
8. **Number of Products** - Products owned (1-4)
9. **Has Credit Card** - 0 (No) or 1 (Yes)
10. **Is Active Member** - 0 (No) or 1 (Yes)

Click **Predict** to get:
- Churn probability (0-1)
- Prediction: "Customer is likely to churn" or "Customer is not likely to churn"

### Training the Model

Open and run the Jupyter notebooks:

```bash
jupyter notebook
```

**Notebooks:**
- `experiments.ipynb` - Main training pipeline
- `hyperparametertuningann.ipynb` - Hyperparameter optimization
- `prediction.ipynb` - Model inference and testing
- `salaryregression.ipynb` - Additional regression analysis

## 📁 Project Structure

```
ANN-Classification-Churn/
│
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Model training & evaluation
├── hyperparametertuningann.ipynb   # Hyperparameter tuning
├── prediction.ipynb                # Prediction pipeline
├── salaryregression.ipynb          # Regression experiments
│
├── requirements.txt                # Python dependencies
│
├── Churn_Modelling.csv            # Dataset (10,000 records)
│
├── model.h5                        # Trained Keras model (65KB)
├── label_encoder_gender.pkl        # Gender encoder
├── onehot_encoder_geo.pkl          # Geography encoder
├── scaler.pkl                      # StandardScaler object
│
└── README.md                       # Documentation
```

## 🔧 Technologies Used

| Technology | Purpose |
|------------|---------|
| **TensorFlow 2.15** | Deep learning framework |
| **Keras** | Neural network API |
| **Streamlit** | Web app framework |
| **scikit-learn** | Preprocessing & metrics |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Matplotlib** | Visualization |
| **TensorBoard** | Training monitoring |
| **Pickle** | Model serialization |

## 📝 Key Implementation Details

### Data Preprocessing

```python
# 1. Label Encoding for Gender
label_encoder_gender = LabelEncoder()
gender_encoded = label_encoder_gender.fit_transform(df['Gender'])

# 2. One-Hot Encoding for Geography
onehot_encoder_geo = OneHotEncoder()
geo_encoded = onehot_encoder_geo.fit_transform(df[['Geography']])

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Model Prediction Flow

```
User Input → Label Encode Gender → One-Hot Encode Geography → 
Scale Features → Neural Network → Sigmoid Output → 
Probability (0-1) → Classification (>0.5 = Churn)
```

## 🚀 Deployment

### Streamlit Cloud

This app is ready for deployment on Streamlit Cloud:

1. Push to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

**Required files:**
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `model.h5` - Trained model
- ✅ `*.pkl` - Encoders and scaler

## 📊 Model Training Process

### Step 1: Data Preparation
- Removed non-predictive columns (RowNumber, CustomerId, Surname)
- Split into features (X) and target (y)
- Train-test split (80-20)

### Step 2: Preprocessing
- Label encoding for binary categorical features
- One-hot encoding for multi-class categorical features
- Standard scaling for numerical features

### Step 3: Model Building
- Sequential model with Dense layers
- ReLU activation for hidden layers
- Sigmoid activation for output
- Dropout for regularization

### Step 4: Training
- Binary crossentropy loss
- Adam optimizer
- Early stopping to prevent overfitting
- TensorBoard for monitoring

### Step 5: Evaluation
- Confusion matrix
- Classification report
- ROC-AUC curve
- Feature importance analysis

## 🔍 Key Insights

### Factors Contributing to Churn

1. **Age** - Older customers (50+) have higher churn rates
2. **Geography** - German customers churn more frequently
3. **Number of Products** - Customers with 3-4 products show elevated churn
4. **Activity Status** - Inactive members are 3x more likely to churn
5. **Account Balance** - Extreme balances (very high or zero) correlate with churn

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Deepak Manola**

- GitHub: [@manola1109](https://github.com/manola1109)
- Repository: [ANN-Classification-Churn](https://github.com/manola1109/ANN-Classification-Churn)

## 🙏 Acknowledgments

- Dataset: Bank customer churn modeling dataset
- Inspiration: Customer retention analytics
- Framework: TensorFlow/Keras documentation
- Deployment: Streamlit Cloud platform

## 📧 Support

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/manola1109/ANN-Classification-Churn/issues)
- Submit a [Pull Request](https://github.com/manola1109/ANN-Classification-Churn/pulls)

---

⭐ **If you found this project helpful, please consider giving it a star!**

**Built with ❤️ using TensorFlow and Streamlit**
