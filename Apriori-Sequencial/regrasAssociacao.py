from efficient_apriori import apriori
import pandas as pd
import sys
sys.path.insert(0, '/home/patricia/Work/PESC/TEBD2')
import tratamento1m
import time

def associacao(arquivo, sup_min, conf_min):

	array_filmes = tratamento1m.tratamento(arquivo)
	inicio = time.time()
	itemsets, rules = apriori(array_filmes, min_support=sup_min,  min_confidence=conf_min)
	
	cont = 0
	listDF = []
	newDf = pd.DataFrame()
	newDf2 = pd.DataFrame()
	
	
	for rule in rules:

		if (len(rule.lhs) == 1 and len(rule.rhs) == 1):

			regra = rule.lhs[0] + " => "+  rule.rhs[0]
			supportStr = "{0:.6f}".format(rule.support)
			confidenceStr = "{0:.6f}".format(rule.confidence)
			liftStr = "{0:.6f}".format(rule.lift)
			nlift = (1 - rule.confidence)/(1 - (rule.confidence/rule.lift))
			nliftstr = "{0:.6f}".format(nlift)
			
			modDfObj = pd.DataFrame({'Regra' : [regra], 'Suporte' : [supportStr], 'Confianca' : [confidenceStr], 'Lift(B,C)' : [liftStr], 'Lift(B,!C)' : [nliftstr]})
			listDF.append(modDfObj)
			newDf2 = pd.concat(listDF)
			cont += 1
	fim = time.time()

	newDf2.to_csv("1M_APRIORI"+"_SUP_" + str(sup_min)+ "_CONF_"+ str(conf_min) +".csv", sep='#', encoding='ISO-8859-1', index = None, header=True)

	print("suporte minimo: " + str(sup_min))
	print("confianca minima: " + str(conf_min))
	print("número de regras: " + str(cont))
	print("Tempo de execução:", fim-inicio)


associacao('ml-1m', 0.30, 0.70)
