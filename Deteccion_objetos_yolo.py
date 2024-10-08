import cv2
from ultralytics import YOLO

# Inicializar el modelo YOLO de ultralytics
model = YOLO('yolov8n.pt')  

# Inicializar captura de video desde la cámara
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Realizar detección de objetos con YOLO
    results = model(frame)

    # Dibujar rectángulos alrededor de los objetos detectados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtener coordenadas del cuadro delimitador
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]  # Nombre del objeto detectado

            # Dibujar el rectángulo y la etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostrar el frame con los objetos detectados
    cv2.imshow('Detección de objetos', frame)

    # Salir con la tecla Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar la cámara y cerrar ventanas
video_capture.release()
cv2.destroyAllWindows()
