# ultralytics para la deteccion de objetos en un video
from ultralytics import YOLO
# cv2 se utiliza para el procesamiento de imagenes y videos
import cv2

#Carga un modelo preentrenado de YOLO desde el archivo
model = YOLO("best.pt")

#Crea un objeto Video Capture para leer el video
cap = cv2.VideoCapture('video.mp4')

#Inicia un bucle infinito para procesar el video fotograma por fotograma.
while True:

    # Lee el siguiente fotograma del video. ret es un booleano que indica si la lectura fue exitosa, y frame es el fotograma leído.
    ret, frame=cap.read()

    #  Utiliza el modelo YOLO para detectar objetos en el fotograma. imgsz=640 especifica el tamaño de la imagen para la detección, y conf=0.20 establece un umbral de confianza del 20% para las detecciones.
    result = model.predict(frame, imgsz=640, conf=0.20 )

    # Genera una imagen con anotaciones basada en los resultados de la detección.
    anotaciones = result[0].plot()

    # Muestra el fotograma anotado en una ventana.
    cv2.imshow('deteccion',anotaciones)



    #Cerrar el programa con la letra q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Libera el objeto VideoCapture.
cap.release()
#Cierra todas las ventanas abiertas por OpenCV.
cv2.destroyAllWindows()