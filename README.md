# grpr-mdl

A basic machine learning model for predicting student grades based on academic performance data.  
This project uses a linear regression model trained on synthetic data that includes study hours, sleep hours, and social media usage.

---

## Features
- Predicts exam scores from academic and lifestyle factors  
- Simple and lightweight ML pipeline  
- Includes synthetic dataset (`student_scores.csv` / `student_scores_500.csv`)  
- Easy to extend with new features or algorithms  

---

## Project Structure

```
├── student\_scores.csv          # Small sample dataset
├── student\_scores\_500.csv      # Large synthetic dataset (500 records)
├── train\_model.py              # Script to train and save the model
├── student\_score.py            # Example script for making predictions
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```


---

## Installation
```bash
git clone https://github.com/your-username/grpr-mdl.git
cd grpr-mdl
pip install -r requirements.txt
````

---

## Usage

### 1. Train the model

```bash
python train_model.py
```

This will train the model on `student_scores.csv` (or `student_scores_500.csv`) and save it as `student_model.pkl`.

### 2. Make predictions

Edit `student_score.py` with your input values:

```python
new_student = pd.DataFrame([[10, 10, 0]], 
    columns=["Hours_Studied", "Sleep_Hours", "Social_Media_Hours"])
```

Then run:

```bash
python student_score.py
```

Example output:

```
Predicted Exam Score: 85.3
```

---

## Requirements

See [requirements.txt](./requirements.txt)

---

## Future Improvements

* Add more features (attendance, assignments, extracurriculars)
* Try other ML algorithms (Random Forest, Gradient Boosting)
* Create a web interface (FastAPI / Flask) for live predictions

---



