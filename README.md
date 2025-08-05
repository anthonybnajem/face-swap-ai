pip install insightface onnxruntime
pip install fastapi uvicorn
pip install opencv-python
pip install python-multipart
pip install requests

# can change location ask gpt
curl -L -o ~/.insightface/models/inswapper_128/inswapper_128.onnx \
"https://drive.usercontent.google.com/download?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF&export=download&confirm=t"


pip install -r requirements.txt

(.venv) anthony@Anthonys-MacBook-Pro insightface % python --version
Python 3.10.12

pip list

Package                Version
---------------------- -----------
albucore               0.0.24
albumentations         1.3.1
annotated-types        0.7.0
certifi                2025.7.14
charset-normalizer     3.4.2
cmake                  4.0.3
coloredlogs            15.0.1
contourpy              1.2.1
cycler                 0.12.1
Cython                 3.1.2
decorator              5.2.1
easydict               1.13
flatbuffers            25.2.10
fonttools              4.59.0
humanfriendly          10.0
idna                   3.10
imageio                2.37.0
imageio-ffmpeg         0.6.0
insightface            0.7
joblib                 1.5.1
kiwisolver             1.4.8
lazy_loader            0.4
matplotlib             3.8.4
moviepy                2.2.1
mpmath                 1.3.0
networkx               3.4.2
numpy                  1.22.0
onnx                   1.18.0
onnxruntime            1.16.2
opencv-python          4.11.0.86
opencv-python-headless 4.11.0.86
packaging              25.0
pillow                 11.3.0
pip                    25.2
prettytable            3.16.0
proglog                0.1.12
protobuf               6.31.1
pydantic               2.11.7
pydantic_core          2.33.2
pyparsing              3.2.3
python-dateutil        2.9.0.post0
python-dotenv          1.1.1
PyYAML                 6.0.2
qudida                 0.0.4
requests               2.32.4
scikit-image           0.22.0
scikit-learn           1.7.1
scipy                  1.11.4
setuptools             65.5.0
simsimd                6.5.0
six                    1.17.0
stringzilla            3.12.5
sympy                  1.14.0
threadpoolctl          3.6.0
tifffile               2025.5.10
tqdm                   4.67.1
typing_extensions      4.14.1
typing-inspection      0.4.1
urllib3                2.5.0
wcwidth                0.2.13


brew install ffmpeg  # Mac
sudo apt install ffmpeg  # Ubuntu/Debian
choco install ffmpeg  # Windows (if using Chocolatey)
