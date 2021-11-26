#deep learning olanından karma yapmalı
import argparse
import os
import networkx
from networkx.drawing.nx_pydot import write_dot
import itertools
import pprint

from keras.models import load_model
from keras import layers
import os


def pecheck(fullpath):
    return open(fullpath).read(2) == "MZ"

def getstrings(fullpath): #aralarındaki benzerlik için stringleri kullanacak
    strings = os.popen("strings '{0}'".format(fullpath)).read()
    strings = set(strings.split("\n"))
    return strings

def jaccard(set1, set2): # bu değiştirilebilir
    intersection = set1.intersection(set2)
    intersection_length = float(len(intersection))
    union= set1.union(set2)
    union_length = float(len(union))
    return intersection_length / union_length

def featurextrandnode(malware_paths, malware_attributes):
    #özellikleri çıkartıp node çizmeliyim
    graph = networkx.Graph()
    for path in malware_paths:
        attributes = getstrings(path)
        print("Extracted {0} attributes from {1} ...".format(len(attributes),path))
        malware_attributes[path]= attributes

        # add each malware file to the graph
        graph.add_node(path,label=os.path.split(path)[-1][:10])
    #verilerini tek tek çıkartıp nodea ekler
    #iterate through all pairs of malware
    for malware1,malware2 in itertools.combinations(malware_paths,2):

        #compute the jaccard distance for the curret pair
        jaccard_index =  jaccard(malware_attributes[malware1],malware_attributes[malware2])

        #if the jaccard distance is above the threshold add an edge 
        if jaccard_index > args.threshold :
            print(malware1,malware2,jaccard_index)
            graph.add_edge(malware1,malware2,penwidth=1+(jaccard_index-args.threshold)*10)
        
    # write the graph to disk so we can visualize it
    write_dot(graph,args.output_dot_file)
    #tüm malwareler ile biizmkini kontrol eder jaccardlarını kıyaslar ona göre bağlantı seçer

#buradaki malware attributes global olmalı
# hem özelliklerini çıkartıp node çiziyor sonra bağlantılarını çiziyor

#burada ekstra bir şey yaptırmamış olurum





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
    parser = argparse.ArgumentParser(
        description="Identify similarities betwooen malware sample and build similarity graph"
    )
    parser.add_argument(
        "target_directory"
        help="Where to save the output graph DOT file"
    )
    parser.add_argument(
        "--jaccard_index_threshold","-j",dest="threshold",type=float,
        default=0.8,help="Threshold above which to create an 'edge' between samples"
    )

    args = parser.parse_args()
    malware_paths=[] # where we'll store the malware file paths
    malware_attributes = dict() # where we'll store the malware strings
   
    for root, dirs, paths in os.walk(args.target_directory):
        # walk the target directory tree and store all of the file paths
        for path in paths :
            full_path = os.path.join(root, path)
            malware_paths.append(full_path)
    
    # PE olup olmadıklarını kontrol eder
    malware_paths = filter(pecheck,malware_paths)



    features_length = 1024
    num_obs_per_epoch = 5000
    batch_size = 128
    # create the model using the function from the model architecture section:
    model = my_model(input_length=features_length)

    #özellikleri çıkartıp node çizme fonksyionu eklenir
    training_generator = featurextrandnode(malware_paths,malware_attributes)
     # and now train the model: son olarak bunlar kullanarak nöronları eğitmek için fit kullanılır
    # bu da 3 parametre alır.  generator  olan oluşturucu parametresi, steps_per_epoch, modelin her dönem işlemesini istediğimiz parti sayısını ayarlar. 
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_obs_per_epoch / batch_size,
        epochs=10,
        verbose=1)
#training generator kullanılarak model eğitimi tamamlanmış olur.

     # save the model modeli böyle direkt kaydedersin
    model.save('my_model.h5')
    # load the model back into memory from the file:
    same_model = load_model('my_model.h5')  # from keras.models.load_model








#oluşturma ve training
#deneme
    

#ilk alacak bunların birbirleri ile olan bağlantılarını çözecek
#sonra  bizim kini alıp validation kısmı olacak ve deniyecek
