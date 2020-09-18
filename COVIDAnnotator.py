#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np


# In[3]:


import re


# In[4]:


path=""


# In[5]:


infile = open(path+'neg_trigs_custom.txt')
nts = []
for line in infile:
    nts.append(line.strip("\t\n"))
    
#nts


# In[6]:



infile = open(path+'COVID-Twitter-Symptom-Lexicon.txt')
symptom_dict = {}
cui_to_name={}
name_to_cui={}

for line in infile:
    splitline=line.split("        ")
    phrase=splitline[2].strip("\t\n").lower()
    cui=splitline[1]
    name=splitline[0]
    symptom_dict[phrase]=cui
    
    cui_to_name[cui]=name
    name_to_cui[name]=cui
    
#symptom_dict


# In[7]:



#cui_dict[cui]=[list of phrases representing that cui]
cui_dict={}

for phrase, cui in symptom_dict.items():
    #make sure the cui is in the dictionary
    #before appending to it
    if not cui in cui_dict:
        cui_dict[cui]=[]
    
    cui_dict[cui].append(phrase)
    
#cui_dict


# In[8]:


#ripped from the Abeed Sarker's lecture notes
#and then modified slightly

import Levenshtein

import itertools
def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) 
    #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window
        
    
def match_dict_similarity(text, expressions, threshold=0.8, verbose=False):
    '''
    :param text:
    :param expressions:
    :return:
    '''
    #threshold = 0.75
    max_similarity_obtained = -1
    best_match = None
    bestPriorTokens=None
    
    #go through each expression
    for exp in expressions:
        
        #create the window size equal to the number of word in the expression in the lexicon
        size_of_window = len(exp.split())
        tokenized_text = list(word_tokenize(text))
        i=0
        for window in run_sliding_window_through_text(tokenized_text, size_of_window):
            window_string = ' '.join(window)

            similarity_score = Levenshtein.ratio(window_string, exp)

            if similarity_score >= threshold:
                priorTokens=tokenized_text[:i]
                if verbose:
                    print (similarity_score,'\t', exp,'\t', window_string)
                if similarity_score>max_similarity_obtained:
                    max_similarity_obtained = similarity_score
                    best_match = window_string
                    bestPriorTokens=priorTokens
                    
            i+=1
        
    priorSentence=''
    if bestPriorTokens!=None:
        for token in bestPriorTokens:
            priorSentence+=(token + " ")
        
    return best_match, priorSentence, max_similarity_obtained

#match_dict_similarity(testtext,testexpressions)


# In[9]:



#check if the part of a sentence that came prior to a symptom
#contains negation
#and if said negation is close enough to the symptom
#to be considered to be apploed to the symptom
def checkNegation(negationPhrases, priorSentence):
    minBetweenWords=None
    negated=False
    negationWord=None
    
    #find the negation (or lack thereof)
    #closest to the symptom
    for negWord in negationPhrases:
        npat=re.compile(r'\b'+negWord+r'\b')
        nmatch_objects = re.finditer(npat,priorSentence)

        for nmatch_object in nmatch_objects:
            #there is a negation word somewhere in here
            #now find how close it is to the symptom
            #find all the words between the negation and the symptom
            betSentence=priorSentence[nmatch_object.end():]
            betWords=word_tokenize(betSentence)

            #find the shortest set of between words
            #that is, find the words between the sympytom
            #and the closest negation word to it
            if minBetweenWords == None or len(betWords) < len(minBetweenWords):
                minBetweenWords=betWords
                negationWord=negWord

    #if any negation words were found, see if they were
    #close enough to the symptom
    if minBetweenWords!=None:
        #how many words can be between the negation and symptom
        #for them to be related?
        bufferRoom=3
        #consider: I don't have a cough or fever
        if 'and' in betWords:
            bufferRoom+=2
        if 'or' in betWords:
            bufferRoom+=2

        if len(minBetweenWords) < bufferRoom:
            negated=True
                
    return negated, negationWord, minBetweenWords

        
        
def annotateSentence(sentenceString, threshold=0.8):
    retCuis=[]
    retFlags=[]
    
    ret = pd.DataFrame()
    
    for cui, expressions in cui_dict.items():
        negated=False
        negWord=None
        betweenWords=None
        
        match, priorSentence, similarity=match_dict_similarity(sentenceString, expressions, threshold=threshold)
        if match!=None:
        
            #now check for negation        
            negated, negWord, betweenWords=checkNegation(nts, priorSentence)
               
            simpExpression=match
            negationFlag=0

            if negated:
                betweenSentence=""
                for word in betweenWords:
                    betweenSentence+=(word+" ")
                simpExpression=negWord +" "+ betweenSentence + simpExpression
                negationFlag=1


            retCuis.append(cui)
            retFlags.append(negationFlag)
            
        
    return retCuis, retFlags


# In[10]:


#converts a list of cuis and flags to their $$$string$$$ versions
def listToString(lst):
    ret=''
    if len(lst)>0:
        ret = "$$$"
        for cui in lst:
            ret = ret + str(cui) + "$$$"
    
    return ret


# In[11]:



def annotatePost(textPost, pid, threshold=0.8):
    
    
    #check if the post is nan (a float)
    if type(textPost)==float:
        return {"ID":pid, "Symptom CUIs":'', "Negation Flag":''}
    
    sentences = sent_tokenize(textPost)
    cuis=[]
    flags=[]

    for sent in sentences:
        #print(sent)
        somecuis, someflags=annotateSentence(sent, threshold=threshold)
        for i in range(len(somecuis)):
            cui = somecuis[i]
            flag=someflags[i]
            
            #if this particular cui is already marked as being in the post, then what?
            if cui in cuis:
                #if the cui is in, then ignore it,
                #unless we're overriding the negation flag
                otherIndex=cuis.index(cui)
                otherFlag=flags[otherIndex]
                
                if otherFlag==0 and flag==1:
                    flags[otherIndex]=1
            else:
                cuis.append(cui)
                flags.append(flag)
        
        
    ann={"ID":pid, "Symptom CUIs":listToString(cuis), "Negation Flag":listToString(flags)}
    
    return ann


# In[12]:


import pandas as pd


def annotateExcelSheet(file, threshold=0.8):
    inputDF=pd.read_excel(file)
    outputDF=pd.DataFrame()
    
    for row in inputDF.iterrows():
        pid=row[1]["ID"]
        post=row[1]['TEXT']
        ann = annotatePost(post, pid, threshold=threshold)
        outputDF=outputDF.append(ann, ignore_index=True)
        
    return outputDF


# In[ ]:
import sys

inputFileName=sys.argv[1]

outputFileName='output'+inputFileName

ann=annotateExcelSheet(inputFileName)

ann.to_excel(outputFileName)


# In[ ]:




