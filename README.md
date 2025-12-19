🧠 Character Recognition using FNN from Scratch (Pure NumPy)

This project implements a Fully Connected Feedforward Neural Network (FNN) from scratch using pure NumPy for recognizing characters represented as binary pixel data.

The goal of this project is learning and understanding neural networks at a fundamental level, without relying on high-level deep learning frameworks such as TensorFlow or PyTorch.

📌 Key Highlights

✅ Neural Network implemented from scratch

✅ Uses only basic Python libraries

✅ No ML frameworks (no TensorFlow / PyTorch / Keras)

✅ Step-by-step forward & backward propagation

✅ Designed for educational clarity

✅ Character recognition using binary image vectors

🧩 Problem Statement

To recognize characters (A–Z, a–z, 0–9) from binary pixel representations using a Feedforward Neural Network, trained and evaluated using NumPy operations.

Each character is represented as a fixed-size binary vector derived from a pixel grid.

📚 Libraries Used
Library	Purpose
numpy	Matrix operations & neural network math
matplotlib	Plotting loss / accuracy graphs
csv	Reading dataset files
collections	Data organization utilities
time	Training time measurement
docx	Exporting results or reports
models.py	Custom utility functions (e.g., CSV reading, helpers)

⚠️ No external ML or DL libraries are used.

🧠 Neural Network Overview

Type: Fully Connected Feedforward Neural Network

Layers:

Input Layer

One or more Hidden Layers

Output Layer

Activation Functions:

Hidden layers → ReLU / Sigmoid

Output layer → Softmax

Loss Function: Cross-Entropy Loss

Optimization: Gradient Descent (manual backpropagation)

🔄 Training Pipeline

Load binary pixel data from CSV

Normalize input values

Initialize weights & biases randomly

Forward propagation

Loss computation

Backpropagation (manual gradient calculation)

Weight updates

Repeat for multiple epochs

Evaluate accuracy on test data

📊 Dataset Description

Stored in CSV format

Each row represents one character

Columns represent:

Binary pixel values (0 or 1)

Corresponding character label

Example:

0,1,1,0,1,0,0,1,...,A

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Dinesh-hubs/Character-Recognition-FNN.git
cd Character-Recognition-FNN

2️⃣ Run Training
python main.py

3️⃣ View Outputs

Training loss curves

Accuracy metrics

Recognition results

🎯 Learning Objectives

This project helps you understand:

How neural networks work internally

Matrix-based forward propagation

Backpropagation from scratch

Weight updates without frameworks

Why modern DL libraries abstract these steps

🧪 Sample Output

Training Loss vs Epoch graph

Final classification accuracy

Correct vs incorrect predictions

📈 Future Improvements

Add noise robustness testing

Extend to CNN (from scratch)

Support real grayscale images

GUI or Web-based drawing interface

Save & load trained weights

🤝 Contributions

Contributions, suggestions, and improvements are welcome!
Feel free to fork this repository and submit a pull request.

📜 License

This project is open-source and intended for educational purposes.

👤 Author

Dinesh
AI & Data Science Enthusiast
Focused on learning deep learning from first principles
