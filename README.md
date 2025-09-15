# 🍷 Wine Quality Predictor using ANN

Predict wine quality with state-of-the-art Artificial Neural Networks.  
Unlock insights from wine data and classify wines with accuracy!

---

## 🚀 Project Highlights

- **End-to-end pipeline:** From data loading to prediction.
- **Easy customization:** Modify model architecture, training options, or add new features.
- **Data visualization:** See relationships and prediction results.
- **Step-by-step instructions:** Get running in minutes!

---

## 📦 Dataset

We use the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)  
Includes physicochemical properties and expert ratings for red/white wines.

---

## 🛠️ Installation & Setup

Follow these steps to get started:

**1. Clone this repository**
```bash
git clone https://github.com/abhinavsai2006/Wine-Qualtity-Predictor-using-ANN.git
cd Wine-Qualtity-Predictor-using-ANN
```

**2. (Optional) Create & activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```
*If `requirements.txt` is missing, manually install:*
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

**4. Download the dataset**
- Obtain `winequality-red.csv` or `winequality-white.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
- Place the CSV file(s) in the project root directory.

---

## 🧑‍💻 Usage

**Train the model:**
```bash
python train.py
```
This script will:
- Load and preprocess data
- Build and train the ANN
- Evaluate performance
- Save and plot results

**Make predictions on new samples:**
```bash
python predict.py
```
Edit `predict.py` to input custom wine features for prediction.

---

## 🧠 Model Details

- **Input:** Physicochemical features of wine
- **Hidden Layers:** Configurable (default: 2 layers, ReLU)
- **Output:** Wine quality score (classification or regression)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy / MSE

You can customize the architecture and hyperparameters in `train.py`.

---

## 📊 Visuals & Results

*Example accuracy, plots, and confusion matrix will appear here after training!*  
You can add your own images by placing them in the repo and referencing like:
```markdown
![Training Accuracy](images/training_accuracy.png)
```

---

## 🤝 Contributing

Questions, suggestions, or improvements are welcome!

1. Fork this repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am "Add my feature"`)
4. Push (`git push origin feature/my-feature`)
5. Open a Pull Request

Feel free to open issues for bug reports or feature requests!

---

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙋 Contact

**Author:** [abhinavsai2006](https://github.com/abhinavsai2006)  
For help, open an issue or contact via GitHub.

---

> _Happy predicting!_ 🍇
