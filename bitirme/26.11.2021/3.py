#deep learning olanından karma yapmalı
import argparse
import os
import networkx
from networkx.drawing.nx_pydot import write_dot
import itertools
import pprint

from keras.models import load_model
import numpy as np
import murmur
import re
from keras import layers
import os


def jaccard(set1, set2): # bu değiştirilebilir
    intersection = set1.intersection(set2)
    intersection_length = float(len(intersection))
    union= set1.union(set2)
    union_length = float(len(union))
    return intersection_length / union_length

def getstrings(fullpath): #aralarındaki benzerlik için stringleri kullanacak
    strings = os.popen("strings '{0}'".format(fullpath)).read()
    strings = set(strings.split("\n"))
    return strings

def pecheck(fullpath):
    return open(fullpath).read(2) == "MZ"


def myfeatureextractall():
    #path alıp özelliklerini çıkartıp node ekleme işini burada yapsın
    



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
    graph = networkx.Graph() 

    for root, dirs, paths in os.walk(args.target_directory):
        # walk the target directory tree and store all of the file paths
        for path in paths :
            full_path = os.path.join(root, path)
            malware_paths.append(full_path)
    
    # PE olup olmadıklarını kontrol eder
    malware_paths = filter(pecheck,malware_paths)

    for path in malware_paths:
        attributes = getstrings(path)
        print"Extracted {0} attributes from {1} ...".format(len(attributes),path)
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
            print malware1,malware2,jaccard_index
            graph.add_edge(malware1,malware2,penwidth=1+(jaccard_index-args.threshold)*10)
        
    # write the graph to disk so we can visualize it
    write_dot(graph,args.output_dot_file)
    #tüm malwareler ile biizmkini kontrol eder jaccardlarını kıyaslar ona göre bağlantı seçer


#ilk alacak bunların birbirleri ile olan bağlantılarını çözecek
#sonra  bizim kini alıp validation kısmı olacak ve deniyecek
