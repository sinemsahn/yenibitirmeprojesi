#modeli eğitmek
# özellikleri çıkarma
# her HTML dosyasını, modelimizin önemli kalıpları hızlı bir şekilde işlemesine ve öğrenmesine olanak tanıyan tek tip boyutlu, sıkıştırılmış bir gösterime dönüştüreceğiz.
#her bir HTML dosyasını 1.024 uzunlukta bir kategori sayıları vektörüne dönüştürüyoruz; burada her kategori sayısı, HTML dosyasındaki hash değeri verilen kategoriye çözümlenen simge sayısını temsil eder. 
import numpy as np
import murmur
import re
import os


def read_file(sha, dir):
	with open(os.path.join(dir, sha), 'r') as fp:
	file = fp.read()
	return file

def extract_features(sha, path_to_files_dir, hash_dim=1024, split_regex=r"\s+"):
	file = read_file(sha=sha, dir=path_to_files_dir)
	tokens = re.split(pattern=split_regex, string=file)
    # büyük bir veri olarak html i olur sonra bunları token olarak böler
    # her bir tokenın hashini alır
    token_hash_buckets = [
	(murmur.string_hash(w) % (hash_dim - 1) + 1) for w in tokens
	]
    # her bir karmanın modulası alınarak kategorilere ayrılır
    token_bucket_counts = np.zeros(hash_dim)
    

    buckets, counts = np.unique(token_hash_buckets, return_counts=True)

    for bucket, count in zip(buckets, counts):
        token_bucket_counts[bucket] = count
        # her kategorideki karma sayısıdır
    return np.array(token_bucket_counts)

# verileri böyle çıkardıktan sonra kerasın bunlarla çalışması lazım
# küçük miktarda veriyle çalışırken model.fit(my_data, my_labels, epochs=10, batch_size=32)

#kendi veri oluşturcumuz

def my_generator(benign_files, malicious_files, path_to_benign_files, path_to_malicious_files, batch_size, features_length=1024):
	n_samples_per_class = batch_size / 2
	assert len(benign_files) >= n_samples_per_class # yeterli veri olup olmadığını kontrol eder
	assert len(malicious_files) >= n_samples_per_class
	while True:
		ben_features = [
			extract_features(sha, path_to_files_dir=path_to_benign_files,
							hash_dim=features_length)
			for sha in np.random.choice(benign_files, n_samples_per_class,
								replace=False)
		]
		mal_features = [
		extract_features(sha, path_to_files_dir=path_to_malicious_files,
		hash_dim=features_length)
		for sha in np.random.choice(malicious_files, n_samples_per_class,
					replace=False)
		]
		all_features = ben_features + mal_features
		labels = [0 for i in range(n_samples_per_class)] + [1 for i in range( n_samples_per_class)]
		idx = np.random.choice(range(batch_size), batch_size)
		all_features = np.array([np.array(all_features[i]) for i in idx])
		labels = np.array([labels[i] for i in idx])
		yield all_features, labels
        # özellikler çıkartılıp etiketlenir


# eğitim verisi oluşturup modeli eğitmek için

batch_size = 128
features_length = 1024
path_to_training_benign_files = 'data/html/benign_files/training/'
path_to_training_malicious_files = 'data/html/malicious_files/training/'
steps_per_epoch = 1000 # artificially small for example-code speed!

train_benign_files = os.listdir(path_to_training_benign_files)
train_malicious_files = os.listdir(path_to_training_malicious_files)

# make our training data generator!
training_generator = my_generator( 
benign_files=train_benign_files,
malicious_files=train_malicious_files,
path_to_benign_files=path_to_training_benign_files,
path_to_malicious_files=path_to_training_malicious_files,
batch_size=batch_size,
features_length=features_length
)
 model.fit_generator(
generator=training_generator,
steps_per_epoch=steps_per_epoch,
epochs=10
)


path_to_validation_benign_files = 'data/html/benign_files/validation/'
path_to_validation_malicious_files = 'data/html/malicious_files/validation/'
# get the validation keys:
val_benign_file_keys = os.listdir(path_to_validation_benign_files)
val_malicious_file_keys = os.listdir(path_to_validation_malicious_files)
# grab the validation data and extract the features:
validation_data = my_generator(
	benign_files=val_benign_files,
	malicious_files=val_malicious_files,
	path_to_benign_files=path_to_validation_benign_files,
	path_to_malicious_files=path_to_validation_malicious_files,
	batch_size=10000,
	features_length=features_length
	 ).next()

model.fit_generator(
	(1) validation_data=validation_data,
	generator=training_generator,
	steps_per_epoch=steps_per_epoch,
	epochs=10
)

# doğrulama verilerini tek seferde belleğe aktarır fit ile yollarız

# modeli kaydetmek ve kullanmak için
from keras.models import load_model
# save the model
(1) model.save('my_model.h5')
# load the model back into memory from the file:
(2)same_model = load_model('my_model.h5')


#modeli değerlendirme
# modeeleri değerlendirmek için daha karmaşık metrikleri auc
#İkili bir öngörücünün doğruluğunu değerlendirmek için kullanışlı bir metriğe eğri altındaki alan (AUC) denir. 
#AUC, Şekil 11-4'te gösterildiği gibi, bu ROC eğrisinin altındaki alanı alarak tüm bu olasılıkları temsil eder.
