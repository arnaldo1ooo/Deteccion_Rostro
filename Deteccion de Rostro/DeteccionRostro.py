import cv2

color_cuadro = (0, 255, 0)
grosor_linea = 3


# Metodo que dibuja cuadro encima de rostro
def dibujar_cuadro_rostro(detectado, imagen, color: tuple, grosor):
    for (x, y, ancho, largo) in detectado:
        cv2.rectangle(
            imagen, (x, y), (x + ancho, y + largo), color, thickness=grosor
        )


# Valida si la cámara se abrió con éxito
def validar_capturador(capturador: any):
    if (capturador.isOpened() == False):
        print("Error al abrir la secuencia de video o el archivo")


capturador_video = cv2.VideoCapture(0)  # Selecciona el primer dispositivo (0) de captura de video

validar_capturador(capturador_video)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while (capturador_video.isOpened()):
    is_exito, frame = capturador_video.read()
    imagen_escala_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte imagen a escala de grises

    # Detecta los rostros de cada frame
    rostros_detectados = face_cascade.detectMultiScale(image=imagen_escala_gris, scaleFactor=1.3, minNeighbors=4)
    dibujar_cuadro_rostro(rostros_detectados, frame, color_cuadro,
                          grosor_linea)  # Dibuja en el face un cuadro con el color deseado

    if is_exito == True:
        cv2.imshow('WebCam Rostros Detected', frame)

        if cv2.waitKey(
                1) == 27:  # Cuando es letra usar cv2.waitKey(25) & 0xFF == ord('q'):, para usar mayusucula o minuscula juntos
            break
    else:
        break

capturador_video.release()
cv2.destroyAllWindows()
