import sklearn
import pathlib
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = pathlib.Path(dataset_path)
        self.y = None
        self.X = None
        self.id_to_class_name_dict = None
        self.landmark_count = 478
        self.landmark_dim = 3
    
    def load_csv(self,file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    def load_landmarks(self, file_path):
        ixyz = self.load_csv(file_path)
        ixyz = np.array(ixyz, dtype=float)
        return ixyz
    
    def load(self):
        print("load start")
        class_dirs = self.dataset_path.iterdir()
        id_name_dir = [(lambda dir : dir.name.split("_")+[dir])(dir) for dir in class_dirs]
        id_name_dir = np.array(id_name_dir)
        self.id_to_class_name_dict = dict(id_name_dir[:,[0,1]])
        # print(self.id_to_class_name_dict)
        
        csv_count = len(list(self.dataset_path.glob('*/*.csv')))
        self.y = np.zeros((csv_count,), dtype=int)
        self.X = np.zeros((csv_count, self.landmark_count*self.landmark_dim), dtype=float)

        # print(csv_count)
        # exit(0)
        csv_count_i = 0
        for class_id, name, dir in id_name_dir:
            # print(class_id, name)
            # print(class_id)
            class_id = int(class_id)
            for csv_path in dir.glob('*.csv'):
                # print(csv_path)
                self.y[csv_count_i] = class_id
                ixyz = self.load_landmarks(csv_path)
                self.X[csv_count_i] = ixyz[:,[1,2,3]].ravel()
                csv_count_i += 1
        # print(self.y)
        self.y = np.identity(len(self.id_to_class_name_dict))[self.y]
        print("load finish")


def train(X,y):
    clf = MLPClassifier(max_iter=10000).fit(X, y)
    return clf
    

def main():
    dataset_loader = DatasetLoader('./tmp')
    dataset_loader.load()
    X = dataset_loader.X
    y = dataset_loader.y
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(y_train)
    # print(y_test)
    # print(np.argmax(y_train, axis=1))
    # print(np.argmax(y_test, axis=1))
    # print(f"train {len(y_train)} test {len(y_test)}")
    clf = train(X_train, y_train)
    # print(clf.predict_proba(X_test))
    # for x , y in zip(X_test,y_test):
    #     print(x.shape)
    #     print(clf.predict(x),y)
    print(clf.predict_proba(X_test), y_test)
    print(clf.score(X_test, y_test))
    
    

if __name__ == '__main__':
    main()