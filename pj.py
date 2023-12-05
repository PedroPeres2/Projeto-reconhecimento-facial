import cv2
import mediapipe as mp
import face_recognition

# Initialize an empty dictionary to store names and encodings
registered_faces = {}

# Load initial registered face images and get their encodings
initial_registered_images = [
    {"name": "Joao Pedro", "image": "rosto1.jpg"},
    {"name": "Vitor Ramos", "image": "rosto2.jpg"},
    {"name": "Rafael", "image": "rosto3.jpg"},
    {"name": "Pedro Peres", "image": "rosto4.jpg"},
]

for entry in initial_registered_images:
    image = face_recognition.load_image_file(entry["image"])
    encoding = face_recognition.face_encodings(image)[0]
    registered_faces[entry["name"]] = encoding

# Initialize OpenCV, Mediapipe, and Face Recognition
webcam = cv2.VideoCapture(1)
face_detection = mp.solutions.face_detection
reconhecedor_rostos = face_detection.FaceDetection()  # Move this line outside of the loop
drawing_utils = mp.solutions.drawing_utils

def recognize_face(face_image):
    unknown_encodings = face_recognition.face_encodings(face_image)

    if len(unknown_encodings) > 0:
        recognized_names = []
        for unknown_encoding in unknown_encodings:
            # Compare the unknown face encoding with the registered faces
            results = face_recognition.compare_faces(list(registered_faces.values()), unknown_encoding, tolerance=0.6)

            # If a match is found, add the corresponding name to recognized_names
            if True in results:
                matched_name = next(name for name, result in zip(registered_faces.keys(), results) if result)
                recognized_names.append(matched_name)
            else:
                recognized_names.append("Unknown")

        return recognized_names
    else:
        return ["Rosto nao reconhecido"]

while True:
    verificador, frame = webcam.read()
    if not verificador:
        break
        
    # Reconhecer os rostos usando Mediapipe
    lista_rostos = reconhecedor_rostos.process(frame)
    
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            # Obter as coordenadas do rosto
            bboxC = rosto.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Verificar se as coordenadas do rosto estão dentro dos limites da imagem
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= iw and y + h <= ih:
                # Recortar a região do rosto
                face_image = frame[y:y+h, x:x+w]
                
                # Convertendo a imagem para o formato correto (RGB) para o face_recognition
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Enviar a região do rosto para a função de reconhecimento facial
                recognized_names = recognize_face(face_image_rgb)
                
                # Desenhar o retângulo ao redor do rosto detectado
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, ', '.join(recognized_names), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Rostos na Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
