import cv2

COLOR_CUADRO_ROSTRO = (0, 255, 0)
COLOR_CUADRO_OJOS = (0, 0, 255)
GROSOR_LINEA_ROSTRO = 2
GROSOR_LINEA_OJOS = 1

# Metodo que dibuja cuadro encima de rostro
def dibujar_cuadro(detectado, imagen, color: tuple, grosor):
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
ojos_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

while (capturador_video.isOpened()):
    is_exito, frame = capturador_video.read()
    imagen_escala_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte imagen a escala de grises

    # Detecta los rostros de cada frame
    rostros_detectados = face_cascade.detectMultiScale(image=imagen_escala_gris, scaleFactor=1.3, minNeighbors=4)
    ojos_detectados = ojos_cascade.detectMultiScale(image=imagen_escala_gris, scaleFactor=1.3, minNeighbors=4)
    dibujar_cuadro(rostros_detectados, frame, COLOR_CUADRO_ROSTRO, GROSOR_LINEA_ROSTRO)  # Dibuja en el face un cuadro con el color deseado
    dibujar_cuadro(ojos_detectados, frame, COLOR_CUADRO_OJOS, GROSOR_LINEA_OJOS)

    if is_exito == True:
        cv2.imshow('WebCam Rostro Ojos Detected', frame)

        if cv2.waitKey(
                1) == 27:  # Cuando es letra usar cv2.waitKey(25) & 0xFF == ord('q'):, para usar mayusucula o minuscula juntos
            break
    else:
        break

capturador_video.release()
cv2.destroyAllWindows()
