import tensorflow as tf

def read_model():
    model = tf.keras.models.load_model('image_rec/SavedModel')
    return model

def predict_image(model, image):

    name_Classes = ['aletria', 'arroz_cabidela', 'baba_camelo', 'bacalhau_bras', 'bacalhau_natas',
    'bola_berlim', 'bolinhos_bacalhau', 'bolo_bolacha', 'cabrito', 'caldo_verde',
    'canja', 'carne_porco_alentejana', 'cozido_portuguesa',
    'esparguete_bolonhesa', 'feijoada', 'francesinha', 'jardineira',
    'leite_creme', 'pao_de_lo', 'pastel_nata', 'rabanada', 'rojoes', 'tarte_maca',
    'tripas_moda_porto']

    img = tf.io.read_file(image) #Reads Image
    img = tf.image.decode_image(img, channels = 3) # decode image into a tensor with 3 color channels
    img = tf.image.resize(img, size = [224, 224]) # reshape img
    img = img/255 #normalize image

    prediction = model.predict(tf.expand_dims(img, axis = 0)) #Make a prediction

    if (len(prediction[0]) > 1):
        pred_classes = name_Classes[prediction.argmax()] #more than one output, takes the maximum
    else:
        pred_classes = name_Classes[int(tf.round(prediction)[0][0])]

    return pred_classes

