from keras.models import load_model
import numpy as np
import murmur
import re
from keras import layers
import os

#modeli eğitmek için özelilkleri önce çıkarmamız lazım

def read_file(sha, dir): 
    #dizin okunur
    with open(os.path.join(dir, sha), 'r') as fp:
        file = fp.read()
    return file


def extract_features(sha, path_to_files_dir,
                     hash_dim=1024, split_regex=r"\s+"):
    # html dosyasını büyük bir dizge olarka okumaya başlar. daha sonra bu dizeyi normal ifadeye dayalı olarak bir dizi jetona böler.
    file = read_file(sha=sha, dir=path_to_files_dir)

    tokens = re.split(pattern=split_regex, string=file)
    # sonra her bir tokenın hashini alır. her bir karmanın modulası alınarak kategorilere ayrılır. 
    token_hash_buckets = [
        (murmur.string_hash(w) % (hash_dim - 1) + 1) for w in tokens
    ]
    #kısaca özellikleri çıkartıp onları hashleyip paketliyor 1024 olarak 
    token_bucket_counts = np.zeros(hash_dim)
    buckets, counts = np.unique(token_hash_buckets, return_counts=True)
    for bucket, count in zip(buckets, counts):
        token_bucket_counts[bucket] = count
    return np.array(token_bucket_counts)

#tüm eğitim verilerini bir kerede işleve aktarmak yerine eğitim verilerini gruplar halinde veriyor. böylece ram tıkanmaz. 

def my_generator(benign_files, malicious_files,
                 path_to_benign_files, path_to_malicious_files,
                 batch_size, features_length=1024):
    n_samples_per_class = batch_size / 2
    assert len(benign_files) >= n_samples_per_class
    assert len(malicious_files) >= n_samples_per_class
    while True:
        # first, extract features for some random benign files:
        ben_features = [
            extract_features(sha, path_to_files_dir=path_to_benign_files,
                             hash_dim=features_length)
            for sha in np.random.choice(benign_files, n_samples_per_class,
                                        replace=False)
        ]
        # now do the same for some malicious files:
        mal_features = [
            extract_features(sha, path_to_files_dir=path_to_malicious_files,
                             hash_dim=features_length)
            for sha in np.random.choice(malicious_files, n_samples_per_class,
                                        replace=False)
        ]
        # concatenate these together to get our features and labels array:
        all_features = ben_features + mal_features
        # "0" will represent "benign", and "1" will represent "malware":
        labels = [0 for i in range(n_samples_per_class)] + [1 for i in range(
            n_samples_per_class)]

        # finally, let's shuffle the labels and features so that the ordering
        # is not always benign, then malware:
        idx = np.random.choice(range(batch_size), batch_size)
        all_features = np.array([np.array(all_features[i]) for i in idx])
        labels = np.array([labels[i] for i in idx])
        yield all_features, labels
# özelliklerini çıkartıp etiketlemelerini yapıyor iyi huylu kötü huylu diye

#modeli eğitme adımları
def make_training_data_generator(features_length, batch_size):
    path_to_training_benign_files = 'data/html/benign_files/training/'
    path_to_training_malicious_files = 'data/html/malicious_files/training/'

    train_benign_files = os.listdir(path_to_training_benign_files)
    train_malicious_files = os.listdir(path_to_training_malicious_files)

    training_generator = my_generator(
        benign_files=train_benign_files,
        malicious_files=train_malicious_files,
        path_to_benign_files=path_to_training_benign_files,
        path_to_malicious_files=path_to_training_malicious_files,
        batch_size=batch_size,
        features_length=features_length
    )
    #bu da gider aldığı pathlerden özellikleri çıkartır etiketler ve döner
    return training_generator


def get_validation_data(features_length, n_validation_files):
    path_to_validation_benign_files = 'data/html/benign_files/validation/'
    path_to_validation_malicious_files = 'data/html/malicious_files/validation/'
    # get the validation keys:
    val_benign_files = os.listdir(path_to_validation_benign_files)
    val_malicious_files = os.listdir(path_to_validation_malicious_files)

    # create the model:
    # grab the validation data and extract the features:
    validation_data = my_generator(
        benign_files=val_benign_files,
        malicious_files=val_malicious_files,
        path_to_benign_files=path_to_validation_benign_files,
        path_to_malicious_files=path_to_validation_malicious_files,
        batch_size=n_validation_files,
        features_length=features_length
    ).next()
    return validation_data


def example_code_with_validation_data(model, training_generator,
                                      steps_per_epoch,
                                      features_length, n_validation_files):
    validation_data = get_validation_data(features_length, n_validation_files)
    model.fit_generator(
        validation_data=validation_data,
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        verbose=1)

    return model
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
    features_length = 1024
    # by convention, num_obs_per_epoch should be roughly equal to the size
    # of your training dataset, but we're making it small here since this
    # is example code and we want it to run fast!
    num_obs_per_epoch = 5000
    batch_size = 128

    # create the model using the function from the model architecture section:
    model = my_model(input_length=features_length)

    # make the training data generator:  eğitim fonksyionu çağırır o da kendi içinde aldığı pathlerden özellikleri çıkartıp etiketleme işini yapan fonksyionu çağırır
    training_generator = make_training_data_generator(
        batch_size=batch_size,
        features_length=features_length
    )
    # and now train the model: son olarak bunlar kullanarak nöronları eğitmek için fit kullanılır
    # bu da 3 parametre alır.  generator  olan oluşturucu parametresi, steps_per_epoch, modelin her dönem işlemesini istediğimiz parti sayısını ayarlar. 
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)
#training generator kullanılarak model eğitimi tamamlanmış olur.

    # now try getting some validation data:
    validation_data = get_validation_data(features_length=features_length,
                                          n_validation_files=1000)
    # and train the model with training and validation data specified: eğitim verilerini kullanarak validation data ile çıkarılanları deniyor
    model.fit_generator(
        validation_data=validation_data,
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)

    # save the model modeli böyle direkt kaydedersin
    model.save('my_model.h5')
    # load the model back into memory from the file:
    same_model = load_model('my_model.h5')  # from keras.models.load_model


#model evalutation kısmına geçmedim ilerde bakılabilir şimdi kendi koduma bakmalı



    