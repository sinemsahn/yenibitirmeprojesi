from keras.models import Model
from keras import layers


def my_model_simple(input_length=1024):
    # basit bir model tanımlaması yapalım shape ile katmanlara bir şekil, dtype ile veri türü ileterek bu modelin bir gözlem için ne tür verileri kabul edeceğini belirtiyoruz 
    input = layers.Input(shape=(input_length,), dtype='float32')
#dense keras modellerini geliştirirken kullanacağınız en yaygın katman türüdür.
# dense iki argüman alır. 513 nöron istediğmizi belirtmek için units ve bu nöronların düzeltilmiş doğrusal birim(ReLU) nöronlar olmasını istediğimizi belirtmek için activiation=relu alır.

    middle = layers.Dense(units=512, activation='relu')(input)
#burada densenin outputunu tanımlıyoruz burada tek bir nöron alır. ve bir sigmoid aktivasyonunu alır. bu çok sayıda veriyi 0 ile 1 arasındaki tek bir skorda birleştirmek için harikadır. 
# çıktı katmanı, girdi olarak nesne, orta katmanımızda ki 512 nöronmuzun çıktılarının hepsinin bu nörona gönderilmesi gerektiğini bildirir.
    output = layers.Dense(units=1, activation='sigmoid')(middle)

    model = Model(inputs=input, outputs=output)
    #en sonda modeli derleriz. optimizer kullanılacak geri yayılım algoritmasının türünü belirtir.
    #loss parametresi eğitim süreci sırasında en aza indirilen şeyi belirtir. 
    #metric eğitim sırasında ve sonrasında model performansını analiz ederken kerasın raporlamasını istediğiniz metriklerin bir listesini iletebilirsiniz.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
#tam bu noktada bu fonksyionu görmek için model.summary()ile oluşturulan modeli görebilirsin.


def my_model(input_length=1024):
    # Note that we can name any layer by passing it a "name" argument.
    input = layers.Input(shape=(input_length,), dtype='float32', name='input')

    # We stack a deep densely-connected network on tops
    x = layers.Dense(1024, activation='relu')(input)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.normalization.BatchNormalization()(x)

    # And finally we add the last (logistic regression) layer:
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    simple_model = my_model_simple(1024)
    model = my_model(1024)
    print(simple_model.summary())