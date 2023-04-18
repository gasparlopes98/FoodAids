import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

def prep_Image(inputImage, img_shape = 224):

    img = tf.io.read_file(inputImage) #Reads Image

    img = tf.image.decode_image(img, channels = 3) # decode image into a tensor with 3 color channels
    img = tf.image.resize(img, size = [img_shape, img_shape]) # reshape img
    img = img/255 #normalize image

    return img

def load_model():
    model = tf.keras.models.load_model('reconhecimento_imagem/SavedModel')
    return model

def predict_and_plot_imgpath_only(model,image_path):

    image = cv2.imread(image_path)

    img = prep_Image(image) #Prepare the image for classification

    prediction = model.predict(tf.expand_dims(img, axis = 0)) #Make a prediction

    if (len(prediction[0]) > 1):
        pred_classes = name_Classes[prediction.argmax()] #more than one output, takes the maximum
    else:
        pred_classes = name_Classes[int(tf.round(prediction)[0][0])]
        
    '''
    #displaying results
    plt.imshow(img)
    plt.title(f"Prediction: {pred_classes}")
    plt.axis(False)
    plt.show()
    '''
    return pred_classes

def predict_and_plot(model, image, class_names):

    img = prep_Image(image) #Prepare the image for classification

    prediction = model.predict(tf.expand_dims(img, axis = 0)) #Make a prediction

    if (len(prediction[0]) > 1):
        pred_classes = name_Classes[prediction.argmax()] #more than one output, takes the maximum
    else:
        pred_classes = name_Classes[int(tf.round(prediction)[0][0])]

    #displaying results
    plt.imshow(img)
    plt.title(f"Prediction: {pred_classes}")
    plt.axis(False)
    plt.show()

    return pred_classes

name_Classes = ['aletria', 'arroz_cabidela', 'baba_camelo', 'bacalhau_bras', 'bacalhau_natas',
 'bola_berlim', 'bolinhos_bacalhau', 'bolo_bolacha', 'cabrito', 'caldo_verde',
 'canja', 'carne_porco_alentejana', 'cozido_portuguesa',
 'esparguete_bolonhesa', 'feijoada', 'francesinha', 'jardineira',
 'leite_creme', 'pao_de_lo', 'pastel_nata', 'rabanada', 'rojoes', 'tarte_maca',
 'tripas_moda_porto']

# Load the saved model from the file
model = tf.keras.models.load_model('reconhecimento_imagem/SavedModel')

#predict_and_plot(model, "C:/Users/diogo/Desktop/Universidade/testes/irm.jpg", name_Classes)

for file in os.listdir("reconhecimento_imagem/testes"):
    predict_img = os.path.join("reconhecimento_imagem/testes", file)
    plt.figure()
    predict_and_plot(model, predict_img, name_Classes)
    plt.show()
