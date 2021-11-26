from keras import layers
from keras.models import Model

input = layers.Input(shape=(1024,),dtype='float32') # katmanlara bir şekil değeri ve bir veri türü bu modelin bir gözlem için ne tür verileri kabul edeceğini belirtiyoruz
# bir boyutta değişken boyutlu girdiler alırsa sayı yerine None (100, None,) olur
middle = layers.Dense(units=512,activation='relu')(input)
# keras modellerini geliştirmek için kullanacağımız en yaygın katman türüdür. Dense iki argüman alır. bu katmanda 512 nöron isteidğimizi belirtir. bu nöronların düzeltilmiş doğrusal birim relu nöronlar olmasını istediğimizi belirtmek için aktivasyon= relu
output = layers.Dense(units=1, activation='sigmoid')(middle)
# tek bir nöron atıyoruz ve bir 'sigmoid' aktivasyon işlevi (6) kullanıyoruz; bu, çok sayıda veriyi 0 ile 1 arasındaki tek bir skorda birleştirmek için harikadır
#Çıkış katmanı, (ortadaki ) girdi olarak nesne, orta katmanımızda ki 512 nöronumuzun çıktılarının hepsinin bu nörona gönderilmesi gerektiğini bildirir.

model = Model(inputs=input, outputs=output)
# yukarıdakileri tek bir model olarak toplar

model.compile(optimizer='adam',
			loss='binary_crossentropy',
			metrics=['accuracy'])
#modeli derlemek eğitmek için kullanılır 3 parametre alır.
# optimizer:  geri yayılım algoritmasının türünü belirtir. Kullanmak istediğiniz algoritmanın adını burada yaptığımız gibi bir karakter dizisi aracılığıyla belirleyebilir veya doğrudan keras.optimizers'dan bir algoritmayı içe aktararak belirli parametreleri algoritmaya aktarabilir veya hatta kendi algoritmanızı tasarlayabilirsiniz.
#loss :  eğitim süreci (geri yayılım) sırasında en aza indirilen şeyi belirtir. Özellikle bu, gerçek eğitim etiketleriniz ile modelinizin öngörülen etiketleri (çıktı) arasındaki farkı temsil etmek için kullanmak istediğiniz formülü belirtir. Yine, bir kayıp işlevinin adını belirtebilir veya keras.losses.mean_squared_error gibi gerçek bir işlevi iletebilirsiniz.
#metrics:  eğitim sırasında ve sonrasında model performansını analiz ederken Keras'ın raporlamasını istediğiniz metriklerin bir listesini iletebilirsiniz. Yine, ['categorical_accuracy', keras.metrics.top_k_categorical_accur cy] gibi dizeleri veya gerçek metrik işlevleri iletebilirsiniz.
# odeli göremk için model.summary() çağır

# daha karmaşık model mimarisi için model_architecture.py yi gör dedi