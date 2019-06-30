
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
# C:\Users\ankunchu\Documents\src\indic_nlp_library\src\indicnlp\script\indic_scripts.py:115: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
#   ALL_PHONETIC_VECTORS= ALL_PHONETIC_DATA.ix[:,PHONETIC_VECTOR_START_OFFSET:].as_matrix()
# C:\Users\ankunchu\Documents\src\indic_nlp_library\src\indicnlp\script\indic_scripts.py:116: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
#   TAMIL_PHONETIC_VECTORS=TAMIL_PHONETIC_DATA.ix[:,PHONETIC_VECTOR_START_OFFSET:].as_matrix()
# C:\Users\ankunchu\Documents\src\indic_nlp_library\src\indicnlp\script\english_script.py:113: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
#   ENGLISH_PHONETIC_VECTORS=ENGLISH_PHONETIC_DATA.ix[:,PHONETIC_VECTOR_START_OFFSET:].as_matrix()

#Try some of the API methods in the Indic NLP library

# Many of the API functions require a language code. We use 2-letter ISO 639-1 codes. Some languages do not have assigned 2-letter codes. We use the following two-letter codes for such languages:
# Konkani: kK
# Manipuri: mP
# Bodo: bD

# 由于不同的输入方法，相同字符的多个表示等，用印度语脚本编写的文本显示了许多古怪的行为。
# 需要规范化文本（canonicalization）的表示，以便NLP应用程序能够以一致的方式处理数据。 规范化主要处理以下问题：
#  - 非间距字符，如ZWJ / ZWNL
#  - 基于Nukta的字母（characters）的 多重表示
#  - 两个部分相关元音符号的多个表示
#  - 键入不一致性：例如 管道（|）用于恶魔病毒
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

input_text = "\u0958 \u0915\u093c"
remove_nuktas=False
factory=IndicNormalizerFactory()
normalizer=factory.get_normalizer('hi',remove_nuktas)
output_text= normalizer.normalize(input_text)

print("input_text = ",input_text)
print("output_text = ",output_text)
print('Length before normalization: {}'.format(len(input_text)))
print('Length after normalization: {}'.format(len(output_text)))