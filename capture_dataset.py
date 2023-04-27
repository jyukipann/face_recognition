import cv2
import mediapipe as mp
import csv
import pathlib
import argparse
import datetime

class DatasetCollector:
    def __init__(self, class_id:int, class_name:str, dataset_dir_path):
        self.__id = class_id
        self.__name = class_name
        self.__dir_path = pathlib.Path(dataset_dir_path)
        self.__save_dir_path = self.__dir_path / f"{self.__id}_{self.__name}"
        self.__save_dir_path.mkdir(parents=True, exist_ok=True)
        print(str(self.__save_dir_path.absolute()))
    
    def __output_csv(self, array, file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(array)
    
    def __output_landmarks_csv(self, landmarks, file_path):
        ixyz = [(lambda i, l: (i, l.x,l.y,l.z))(i, l) for i, l in enumerate(landmarks.landmark)]
        self.__output_csv(ixyz, file_path)
    
    def save_landmarks(self, landmarks):
        date_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_file_path = self.__save_dir_path / f"{self.__id}_{self.__name}_{date_now}.csv"
        print(str(save_file_path))
        self.__output_landmarks_csv(landmarks, save_file_path)
        
        
        
class MediapipeFaceMesh:
    def __init__(self):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_face_mesh = mp.solutions.face_mesh
        self.__mp_face_mesh_model = self.__mp_face_mesh.FaceMesh(
            # static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,)
        self.image = None
        self.annotated_image = None
        self.landmarks = None
        
    def __annotate_image(self):
        if not self.landmarks.multi_face_landmarks:
            self.annotated_image = self.image
            return
        annotated_image = self.image.copy()
        face_landmarks = self.landmarks.multi_face_landmarks[0]
        self.__mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.__mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.__mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        self.__mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.__mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.__mp_drawing_styles
            .get_default_face_mesh_contours_style())
        self.__mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=self.__mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.__mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        self.annotated_image = annotated_image
        
    def inference(self, image):
        self.image = image
        self.landmarks = self.__mp_face_mesh_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.__annotate_image()
        return self.landmarks
    
    def __del__(self):
        if self.__mp_face_mesh_model is not None: 
            self.__mp_face_mesh_model.close()

def main(args):
    dataset_collector = DatasetCollector(args.class_id, args.class_name, args.dataset_dir)
    face_mesher = MediapipeFaceMesh()
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        face_mesher.inference(image)
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(face_mesher.annotated_image, 1))
        key = cv2.waitKey(5)
        if key & 0xFF == 27:
            break
        if key & 0xFF == ord('s'):
            if face_mesher.landmarks.multi_face_landmarks:
                dataset_collector.save_landmarks(face_mesher.landmarks.multi_face_landmarks[0])
                pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='capture_dataset',
        description='capture face landmarks and save it as csv',
        epilog='')
    
    parser.add_argument('class_id')
    parser.add_argument('class_name')
    parser.add_argument('dataset_dir')
    args = parser.parse_args()
    
    main(args)
