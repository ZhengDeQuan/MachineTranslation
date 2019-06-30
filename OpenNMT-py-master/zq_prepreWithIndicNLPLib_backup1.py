'''
like zq_prepreprocess.py
This file doing the prepreprocess work.
But when it comes to the tokenization procedure, this script would use the Indic_NLP_Library
to tokenize the sentence into words.
this script reads in the HindiEng/
outputs the WordSegementHindiEng/
'''
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME = r"/home/nemo/zhengquan/indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES = r"/home/nemo/zhengquan/indic_nlp_resources"

#Add Library to Python Path

import sys
sys.path.append(r'{}/src'.format(INDIC_NLP_LIB_HOME))
# Export environment variable
#   export INDIC_RESOURCES_PATH=<path>
#OR
from indicnlp import common
common.set_resources_path(INDIC_NLP_RESOURCES)

#Initialize the Indic NLP library

from indicnlp import loader
loader.load()

#Tokenization
from indicnlp.tokenize import indic_tokenize

indic_string='अनूप,अनूप?।फोन'

print('Input String: {}'.format(indic_string))
print('Tokens: ')
for t in indic_tokenize.trivial_tokenize(indic_string):
    print(t)

from indicnlp.morph import unsupervised_morph
from indicnlp import common

# This step will call the service which is very slow


indic_string='आपल्या हिरड्यांच्या आणि दातांच्यामध्ये जीवाणू असतात .'+'अनूप,अनूप?।फोन'
indic_res1 = indic_tokenize.trivial_tokenize(indic_string)
print(type(indic_res1))

print("indic_res1 = ",indic_res1)
#Word Segmentation
from indicnlp.morph import unsupervised_morph
from indicnlp import common

# This step will call the service which is very slow
analyzer=unsupervised_morph.UnsupervisedMorphAnalyzer('mr')
print("o1")
# analyzes_tokens=analyzer.morph_analyze_document(indic_string.split(' '))
# after the step above , this step is much faster
analyzes_tokens=analyzer.morph_analyze_document(indic_res1)
print(type(analyzes_tokens))
for w in analyzes_tokens:
    print(w)



#Transliteration
#import json
import requests
from urllib.parse import  quote

print("---"*10)
text=quote('मनिश् जोए')
text=quote('ब्लुए चोलोउर् कि द्रेस्स् बहुत् अच्ह लग् रह है बहुत् ज्यद')

print(text)
url='http://www.cfilt.iitb.ac.in/indicnlpweb/indicnlpws/transliterate_bulk/hi/en/{}/rule'.format(text)
print(url)
response = requests.get(url)
print(response)
print(response.json())

print("---"*10)

text=quote('manish, joe')
print(text)
url='http://www.cfilt.iitb.ac.in/indicnlpweb/indicnlpws/transliterate_bulk/en/hi/{}/rule'.format(text)
print(url)
response = requests.get(url)
print(response)
print(response.json())
# #Machine Translation
#
# import json
# import requests
# from urllib.parse import  quote
#
# text=quote('Mumbai is the capital of Maharashtra')
# # text=quote('मनिश् जोए')
# url='http://www.cfilt.iitb.ac.in/indicnlpweb/indicnlpws/translate/en/mr/{}/'.format(text)
# ## Note the forward slash '/' at the end of the URL. It's should be there, but please live with it for now!
#
# print(url)
# response = requests.get(url)
# print(response)
# # print(response.text)
# print(response.json())
