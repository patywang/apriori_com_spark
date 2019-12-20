import os
import shutil
import sys
import time

from pyspark import SparkContext


def gerar_proximo_c(f_k, k):
    next_c = [var1 | var2 for index, var1 in enumerate(f_k) for var2 in f_k[index + 1:] if
              list(var1)[:k - 2] == list(var2)[:k - 2]]
    
    return next_c


def gerar_f_k(sc, c_k, itemset_comp, sup):
    def get_sup(x):
        x_sup = len([1 for t in itemset_comp.value if x.issubset(t)])
        if x_sup >= sup:
            return x, x_sup
        else:
            return ()

    f_k = sc.parallelize(c_k).map(get_sup).filter(lambda x: x).collect()
    return f_k

def apriori(sc, f_input,f_input2, f_output, min_sup,min_conf):
    # le as cestas
    data = sc.textFile(f_input)
    #gera as cestas dado o arquivo de ratings
    itemset = data.map(lambda x: x.split('::')).map(lambda x: (int(x[0]),int(x[1]))).groupByKey().map(lambda x: list(set(x[1])))
    #count do total de cestas
    n_amostras = itemset.count()
    
    # suporte minimo
    sup = n_amostras * min_sup
    
    # compartilha o itemset com todos os workers
    itemset_comp = sc.broadcast(itemset.map(lambda x: set(x)).collect())
    
    # guarda para todos os freq_k
    frequent_itemset = []

    #prepara o candidato_1
    k = 1
    c_k = itemset.flatMap(lambda x: set(x)).distinct().collect()
    
    c_k = [{x} for x in c_k]
    

    # quando candidate_k nao esta vazio
    while len(c_k) > 0:
        # gera freq_k
        f_k = gerar_f_k(sc, c_k, itemset_comp, sup)

        frequent_itemset.append(f_k)
        # gera o candidato_k+1
        k += 1
        c_k = gerar_proximo_c([set(item) for item in map(lambda x: x[0], f_k)], k)

    #salva os resultados no sistema de arquivos
    dtCestas = sc.parallelize(frequent_itemset, numSlices=1)
    
    
    sets_1 ={}
    sets_2={}
    for cestas in dtCestas.collect():
        for rule in cestas:
            if(len(list(rule[0])) == 1):
               sets_1[rule[0].pop()] = rule[1]
            elif (len(rule[0]) == 2):
                sets_2[str(rule[0].pop())+ '-' + str(rule[0].pop())] = rule[1]
    


    data2 = sc.textFile(f_input2)
    mv = data2.map(lambda line2: [item for item in line2.strip().split('::')]).collect()
      
    movies=dict()
       
    for movie in mv:
        movies[int(movie[0])]=movie[1]   
    rules=[]
    for sets in sets_2.keys():
        str_aux=''
        antecessor  = sets.split('-')[0]
        consequente = sets.split('-')[1]
        confidence = sets_2[sets] / float(sets_1[int(antecessor)])
        str_aux = movies[int(antecessor)] +' -> '+ movies[int(consequente)]+' conf : ({0:.4f})'.format(confidence)
        
        if(confidence >= min_conf):
            rules.append(str_aux)
    
    for r in rules:
        print("######",r)

    
if __name__ == "__main__":
    print("ENTROU AQUI NO MAIN")
    ini = time.time()
    apriori(SparkContext(appName="Spark Apriori"), "/home/patricia/Work/PESC/TEBD2/Apriori-Spark/data/ratings_1mb.dat","/home/patricia/Work/PESC/TEBD2/Apriori-Spark/data/movies.dat", "/home/patricia/Work/PESC/TEBD2/Apriori-Spark/data/result/1mb", 0.30,0.7)
    fim = time.time()
    print ("##########tempo total ", fim-ini)