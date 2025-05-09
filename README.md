## Important Instructions

### FOR UBUNTU

#### 1) Terminal:
```bash
sudo apt update
sudo apt install build-essential
```

This installs gcc, g++  
Check version:
```bash
g++ --version
```

#### 2) Write your code with filename `test.cpp`

#### 3) Compile with OpenMP flag:
```bash
g++ -fopenmp test.cpp -o test
```

#### 4) Run:
```bash
./test
```

---

#### उद्या जर कोणाला DL चा practical आला तर:
#### टर्मिनलमध्ये खालील कमांड्स step by step टाका:

```bash
sudo apt update  
sudo apt install python3-venv  
python3 -m venv tfenv  
source tfenv/bin/activate  
pip install --upgrade pip  
pip install tensorflow  
pip install notebook  
pip install scikit-learn  
```

यानंतर ही कमांड टाका:

```bash
jupyter notebook
```

यामुळे ब्राऊझरमध्ये नोटबुक उघडेल. तिथे new file उघडून code करा.

#### जर पुन्हा error आला तर नवीन टर्मिनल उघडा आणि:

```bash
source tfenv/bin/activate
```

#### त्यानंतर:

```bash
pip install pandas
pip install matplotlib
pip install tensorflow
```

या कमांड्स टाकून आवश्यक त्या लायब्ररी इंस्टॉल करा.

---

### If notebook isn't installed directly try installing jupyterlab first:

```bash
sudo apt update  
sudo apt install python3-venv  
python3 -m venv tfenv  
source tfenv/bin/activate  
pip install --upgrade pip  
pip install tensorflow jupyterlab notebook scikit-learn
