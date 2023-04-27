# face_recognition 顔認証テスト
## installation
```cmd.exe
conda create -n face_recognition python=3.8
conda activate face_recognition
pip install mediapipe
pip install scikit-learn
```

## 方針
- mediapipe face meshのランドマークを特徴量としてコサイン類似度で分類
- ランドマークをNNに入れてごり押し
    - 採用

## how to use
1. データセットの収集
    - python capture_dataset.py class_id class_name dataset_dir
1. 学習＆推論
    - python face_classfier.py
