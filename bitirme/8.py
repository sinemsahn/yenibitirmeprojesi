#jaccard indexine göre karar verdiğinden float32 olur yine
from keras.models import load_model
import os
#model.h5 olarak kaydetmiş olduğu modeli sonradan matploit ile görselleştirmesini yaparsın

import argparse
import os
import networkx
from networkx.drawing.nx_pydot import write_dot

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
    #modeli derliyoruz
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    #ile modelin katmanını basar
    return model

#model yapısında herhangi bir dğeişiklik yapmadan direkt olarak aldım ama bence tekrar bir incelenmeli değişiklik neler yapılmalı
#modelin mimarisini ve veri akışını tanımlar.
#eğitilmeye hazır bir keras modeli elde ederiz. eğitmemiz ve doğrulama verieri üzerinde test etmemiz lazım
#eğitim içi gerekli adımları ve verileri yollayacağız
#verilerden bazıları seçer doğrulama klasörüne atar bunlarıda test etmesi için veririz her test etmesinde bize bir çıktı basmasını söyleriz

#burada ise benzerlik jaccard indexinden ortakları bulmasını isteyeceğiz

def make_training_data_generator(malware_paths):
    for path in malware_paths:
        attributes = getstrings(path)
        #print("Extracted {0} attributes from {1} ...".format(len(attributes),path))
        malware_attributes[path]= attributes

        # add each malware file to the graph
        graph.add_node(path,label=os.path.split(path)[-1][:10])
    return malware_attributes

#bu fonksiyondan ../datayı alıp buradakilerden jaccard indexlerini hesaplayıp node oluşturmalı

def jaccard(set1, set2): 
    intersection = set1.intersection(set2)
    intersection_length = float(len(intersection))
    union= set1.union(set2)
    union_length = float(len(union))
    return intersection_length / union_length

def getstrings(fullpath): 
    strings = os.popen("strings '{0}'".format(fullpath)).read()
    strings = set(strings.split("\n"))
    return strings

def pecheck(fullpath):
    f  = open(fullpath)
    start = f.read(2)
    return start == 'MZ'
    #return open(fullpath).read(2)  ==  "MZ"
    #return 1

def get_validation_data(malware1, malware_paths,output_dot_file):
    for malware2 in malware_paths:
        #compute the jaccard distance for the curret pair
        jaccard_index =  jaccard(malware_attributes[malware1],malware_attributes[malware2])

        #if the jaccard distance is above the threshold add an edge 
        if jaccard_index > args.threshold :
            print("benzerlik var" + malware1 +" ile "+malware2 +" oranı  "+str(jaccard_index))
            graph.add_edge(malware1,malware2,penwidth=1+(jaccard_index-args.threshold)*10)
        
    # write the graph to disk so we can visualize it
    write_dot(graph,output_dot_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Identify similarities betwooen malware sample and build similarity graph"
    )
    parser.add_argument(
        "target_directory",
        help="Directory containing malware"
    )
    parser.add_argument(
        "deneme_malware_directory",
        help="bakilacak malware yolu"
    )
    parser.add_argument(
         "output_dot_file",
	help="Wehre to save the output graph DOT file"
    )
    parser.add_argument(
        "--jaccard_index_threshold","-j",dest="threshold",type=float,
        default=0.8,help="Threshold above which to create an 'edge' between samples"
    )

    args = parser.parse_args()
    malware_paths=[] # where we'll store the malware file paths
    malware_attributes = dict() # where we'll store the malware strings
    graph = networkx.Graph() 

    for root, dirs, paths in os.walk(args.target_directory):
        # walk the target directory tree and store all of the file paths
        for path in paths :
            full_path = os.path.join(root, path)
            malware_paths.append(full_path)
    
    # filter out any pahs that arent PE files
    malware_paths = filter(pecheck,malware_paths)


    features_length = 1024
    # by convention, num_obs_per_epoch should be roughly equal to the size
    # of your training dataset, but we're making it small here since this
    # is example code and we want it to run fast!
    num_obs_per_epoch = 5000
    batch_size = 128
    
    # create the model using the function from the model architecture section:
    model = my_model(input_length=features_length)

    # make the training data generator:
    training_generator = make_training_data_generator(
        malware_paths
    )
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)
    #verileri gönderip çıkartıp öğrenmiş olacak
    malware1= args.deneme_malware_directory
    validation_data = get_validation_data(malware1, malware_paths,args.output_dot_file)
    # and train the model with training and validation data specified:
    model.fit_generator(
        validation_data=validation_data,
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)
#validation data olarak jaccard büyük olanları döndürürüz

    # save the model
    model.save('my_model.h5')
    # load the model back into memory from the file:
    same_model = load_model('my_model.h5')  # from keras.models.load_model






