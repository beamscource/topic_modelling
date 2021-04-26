# Author Eugen Klein, February 2021

import pandas as pd
from tqdm import tqdm
import numpy as np
import re

df = pd.read_excel(r'all_transcripts.xlsx', sheet_name='Data', engine='openpyxl')
df['transcript'] = df['request'].str.lower().str.replace(r'[^\w\s]+', '', regex=True).str.strip()

######## BRIEF CORPUS OVERVIEW
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

transcripts_ru = df.loc[df['language'] == 'ru']['request'].tolist()
transcripts_ukr = df.loc[df['language'] == 'ukr']['request'].tolist()

# Russian
word_count_transcripts_ru = [len(request) for request in transcripts_ru if isinstance(request, str)]
transcripts_ru_stats = stats.describe(word_count_transcripts_ru)
mean_count_transcripts_ru = stats.describe(word_count_transcripts_ru)[2]
# 37.842039079125556
num_transcripts_ru = stats.describe(word_count_transcripts_ru)[0]
# 10338

# Ukrainian
word_count_transcripts_ukr = [len(request) for request in transcripts_ukr if isinstance(request, str)]
transcripts_ukr_stats = stats.describe(word_count_transcripts_ukr)
mean_count_transcripts_ukr = stats.describe(word_count_transcripts_ukr)[2]
# 45.333539221741816
num_transcripts_ukr = stats.describe(word_count_transcripts_ukr)[0]
# 8095

# plot word count distribution
plot = sns.distplot(word_count_transcripts_ru, norm_hist=False, kde=False, bins=100)
plot.set(xlabel='Word count Russian', ylabel='Frequency')
plt.show(block=False)

plot = sns.distplot(word_count_transcripts_ukr, norm_hist=False, kde=False, bins=100)
plot.set(xlabel='Word count Ukrainian', ylabel='Frequency')
plt.show(block=False)

######### REPLACING NUMERAL + WORD INTO LEMMATALIZED FORM
# single digits are exluded later during lemmatization

# command to search for pattern and count them
# df['transcript'].str.contains('грн\b').value_counts()
# df['transcript'][df['transcript'].str.contains('\\d+грн', na=False)]
# df['transcript'][df['transcript'].str.contains('\\d+грн', na=False)].value_counts()

df['transcript'][df['transcript'].duplicated(na=False)]

# define dict with pattern/replacement
regex_dict = {r'\d+гр\b':r'гривна'
                , r'\d+грн\b':r'гривна'
                , r'\d+ гр\b':r'гривна'
                , r'\d+ грн\b':r'гривна'
                , r'\d+гривен\b':r'гривна'
                , r'\d+гривень\b':r'гривна'
                , r'\d+ гривен\b':r'гривна'
                , r'\d+ гривень\b':r'гривна'
                , r'\bгр\b':r'гривна'
                , r'\bгрн\b':r'гривна'
                , r'\d+числа\b':r'число'
                , r'\d+ числа\b':r'число'
                , r'\d+число\b':r'число'
                , r'\d+ число\b':r'число'
                , r'\d+мб\b':r'мегабайт'
                , r'\d+ мб\b':r'мегабайт'
                , r'\d+мегабайт\b':r'мегабайт'
                , r'\d+ мегабайт\b':r'мегабайт'
                , r'\d+мегабит\b':r'мегабайт'
                , r'\d+ мегабит\b':r'мегабайт'
                , r'\d+мегабай\b':r'мегабайт'
                , r'\d+ мегабай\b':r'мегабайт'
                , r'\d+мегобай\b':r'мегабайт'
                , r'\d+ мегобай\b':r'мегабайт'
                , r'\d+мегобайт\b':r'мегабайт'
                , r'\d+ мегобайт\b':r'мегабайт'
                , r'\bмб\b':r'мегабайт'
                , r'\d+гб\b':r'гигабайт'
                , r'\d+ гб\b':r'гигабайт'
                , r'\d+гигабайт\b':r'гигабайт'
                , r'\d+ гигабайт\b':r'гигабайт'
                , r'\d+гигабит\b':r'гигабайт'
                , r'\d+ гигабит\b':r'гигабайт'
                , r'\d+гигибайт\b':r'гигабайт'
                , r'\d+ гигибайт\b':r'гигабайт'
                , r'\d+гигобайт\b':r'гигабайт'
                , r'\d+ гигобайт\b':r'гигабайт'
                , r'\bгб\b':r'гигабайт'
                , r'\d+мг\b':r'мегабайт'
                , r'\d+ мг\b':r'мегабайт'
                , r'\bмг\b':r'мегабайт'
                , r'\bинет\b':r'интернет'
                , r'\bінет\b':r'интернет'
                , r'\bінтернет\b':r'интернет'
                , r'\bwi fi\b':r'wifi'
                , r'\bтб\b':r'тв'
                , r'\bдоп\b':r'дополнительно'
                , r'\bул\b':r'улица'
                , r'\bобл\b':r'область'
                , r'\b4 g\b':r'4g'
                , r'\b3 g\b':r'3g'
                , r'\bне ':r'не_'
                , r'\n':r''}

for pattern, replacement in tqdm(regex_dict.items(), total=len(regex_dict.items())):
    df['transcript']=df['transcript'].str.replace(pattern,replacement,regex=True)

# split data fram based on language tag
df_ru = df.loc[df['language'] == 'ru']
df_ukr = df.loc[df['language'] == 'ukr']

transcripts_list_ru = df.loc[df['language'] == 'ru']['transcript'].tolist()
transcripts_list_ukr = df.loc[df['language'] == 'ukr']['transcript'].tolist()

np.save(r'transcripts_list_ru.npy', transcripts_list_ru)
np.save(r'transcripts_list_ukr.npy', transcripts_list_ru)
transcripts_list_ru = np.load(r'transcripts_list_ru.npy', allow_pickle=True)
transcripts_list_ukr = np.load(r'transcripts_list_ukr.npy', allow_pickle=True)
transcripts_list_ru = transcripts_list_ru.tolist()
transcripts_list_ukr = transcripts_list_ukr.tolist()

######### LEMMATALIZATION AND REPLACEMENTS RUSSIAN

import pymorphy2

morph_ru = pymorphy2.MorphAnalyzer()
morph_ukr = pymorphy2.MorphAnalyzer(lang='uk')

max_len_lemma=15

# exclude modal verbs and frequent nouns
with open(r'stopwords-ru.txt', 'r', encoding='utf-8') as f:
    stopwords_ru = f.readlines()

with open(r'stopwords-ukr.txt', 'r', encoding='utf-8') as f:
    stopwords_ukr = f.readlines()

stopwords_ru = [word.strip() for word in stopwords_ru]
stopwords_ukr = [word.strip() for word in stopwords_ukr]

with open(r'keepwords-ru.txt', 'r', encoding='utf-8') as f:
    keepwords_ru = f.readlines()

with open(r'keepwords-ukr.txt', 'r', encoding='utf-8') as f:
    keepwords_ukr = f.readlines()

keepwords_ru = [word.strip() for word in keepwords_ru]
keepwords_ukr = [word.strip() for word in keepwords_ukr]

pos_to_keep = ['NOUN', 'VERB', 'ADJF', 'PRED', 'ADVB', 'CONJ', 'PRCL']
def lemmatize_txt(transcript_list, stopwords, keepwords, morph):
    #digits = re.compile('[0-9a-z]')
    digits = re.compile('[0-9]') # keep latin words
    lemmatized_transcript_list = []
    for i, txt in enumerate(tqdm(transcript_list)):
        lemmatazed_txt = []
        for word in txt.split():
            word = word.strip()
            if word in stopwords or len(word) > max_len_lemma: # extra for Ukrainian since lemma dict is worse than for Russian
                continue
            if word in keepwords:
                lemmatazed_txt.append(word)
                continue
            lemma = morph.parse(word)[0].normal_form
            if re.search(digits, lemma):
                lemma = re.sub(digits, lemma, '')
            if lemma in stopwords or len(lemma) > max_len_lemma:
                continue
            if 'Name' not in morph.parse(word)[0].tag: # get rid of names
                if morph.parse(word)[0].tag.POS in pos_to_keep or 'LATN' in  morph.parse(word)[0].tag: # keep latin terms (sim, app etc.)
                    lemmatazed_txt.append(lemma)
        lemmatazed_txt = ' '.join(lemmatazed_txt)
        lemmatized_transcript_list.append(lemmatazed_txt)
    return lemmatized_transcript_list

# run lemmatizer
lemm_transcripts_list_ru = lemmatize_txt(transcripts_list_ru, stopwords_ru, keepwords_ru, morph_ru)
lemm_transcripts_list_ukr = lemmatize_txt(transcripts_list_ukr, stopwords_ukr, stopwords_ukr, morph_ukr)

np.save(r'lemm_transcripts_list_ru.npy', lemm_transcripts_list_ru)
np.save(r'lemm_transcripts_list_ukr.npy', lemm_transcripts_list_ru)
lemm_transcripts_list_ru = np.load(r'lemm_transcripts_list_ru.npy', allow_pickle=True)
lemm_transcripts_list_ukr = np.load(r'lemm_transcripts_list_ukr.npy', allow_pickle=True)
lemm_transcripts_list_ru = lemm_transcripts_list_ru.tolist()
lemm_transcripts_list_ukr = lemm_transcripts_list_ukr.tolist()

from scipy import stats
# average number of lemmata per transcript for Russian
word_count_transcript_ru = [len(request) for request in lemm_transcripts_list_ru if isinstance(request, str)]
transcript_ru_stats = stats.describe(word_count_transcript_ru)[2]
# 22.54124414976599

# average number of lemmata per transcript for Ukrainian
word_count_transcript_ukr = [len(request) for request in lemm_transcripts_list_ukr if isinstance(request, str)]
transcript_ukr_stats = stats.describe(word_count_transcript_ukr)[2]
# 27.956218662101016

# extract verb and noun lists to expand ukr lexicon used for language separation
def extract_lexicon(transcript_list):
    #digits = re.compile('[0-9a-z]') # ignore digits (dates, money and MB amounts)
    digits = re.compile('[0-9]')
    verb_list = []
    noun_list = []
    adjv_list = []
    for i, txt in enumerate(tqdm(transcript_list)):
        for word in txt.split():
            word = word.strip()
            lemma = morph.parse(word)[0].normal_form
            # get rid of lemmata that contain digits and latin characters
            if re.search(digits, lemma):
                continue
            if 'Name' not in morph.parse(word)[0].tag: # get rid of names
                if morph.parse(word)[0].tag.POS == 'NOUN':
                    if lemma not in noun_list and len(lemma) < 15: # get rid of too long lemmata
                        noun_list.append(lemma)
                if morph.parse(word)[0].tag.POS == 'VERB':
                    if lemma not in verb_list and len(lemma) < 15:
                        verb_list.append(lemma)
                #if morph.parse(word)[0].tag.POS == 'ADJF' or morph.parse(word)[0].tag.POS == 'PRED':
                else:
                    if lemma not in adjv_list and len(lemma) < 15:
                        adjv_list.append(lemma)
        verb_list.sort()
        noun_list.sort()
        adjv_list.sort()
    return verb_list, noun_list, adjv_list

# Russian
ru_verbs, ru_nouns, ru_adjv = extract_lexicon(transcripts_list_ru)

with open(r'verbs_lemmata_ru.txt', 'w', encoding='utf-8') as f:
    for line in ru_verbs:
        f.writelines(line + '\n')

with open(r'nouns_lemmata_ru.txt', 'w', encoding='utf-8') as f:
    for line in ru_nouns:
        f.writelines(line + '\n')

with open(r'adjvs_lemmata_ru.txt', 'w', encoding='utf-8') as f:
    for line in ru_adjv:
        f.writelines(line + '\n')

# Ukrainian
ukr_verbs, ukr_nouns, ukr_adjv = extract_lexicon(transcripts_list_ukr)

with open(r'verbs_lemmata_ukr.txt', 'w', encoding='utf-8') as f:
    for line in ukr_verbs:
        f.writelines(line + '\n')

with open(r'nouns_lemmata_ukr.txt', 'w', encoding='utf-8') as f:
    for line in ukr_nouns:
        f.writelines(line + '\n')

with open(r'adjvs_lemmata_ukr.txt', 'w', encoding='utf-8') as f:
    for line in ukr_adjv:
        f.writelines(line + '\n')

############ Tfidf vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
import operator

vectorizerTf = TfidfVectorizer(
    max_df=0.95, # percentage of documents a term has to appear in
    min_df=2, # percentage (count when integers) of documents a term has to appear in 
    # max_df=.5,
    # min_df=10,
    max_features=None,
    ngram_range=(1, 3),
    norm=None,
    binary=True,
    use_idf=False,
    sublinear_tf=False
)

# Russian
tfidf_ru = vectorizerTf.fit_transform(lemm_transcripts_list_ru)
vocab_ru = vectorizerTf.get_feature_names()
print(len(vocab_ru))
# 5394

# Ukrainian
tfidf_ukr = vectorizerTf.fit_transform(lemm_transcripts_list_ukr)
vocab_ukr = vectorizerTf.get_feature_names()
print(len(vocab_ukr))
# 5598

# visualize Tfidf scores per lemma
def rank_terms(matrix, terms):
    # get the sums over each column
    sums = matrix.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documents
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

# Russian
lemma_tfidf_ru_scores = pd.DataFrame(rank_terms(tfidf_ru, vocab_ru))
lemma_tfidf_ru_scores.columns=['lemma','Tfidf_score']
lemma_tfidf_ru_scores.sort_values('Tfidf_score', ascending=False)

# Ukrainian
lemma_tfidf_ukr_scores = pd.DataFrame(rank_terms(tfidf_ukr, vocab_ukr))
lemma_tfidf_ukr_scores.columns=['lemma','Tfidf_score']
lemma_tfidf_ukr_scores.sort_values('Tfidf_score', ascending=False)

# save lemmata with their tfidf scores: Russian
with open(r'lemmata_tfidf_ru.txt', 'w', encoding='utf-8') as f:
    space = 30
    gap = space - len('lemma')
    f.writelines('lemma' + ' '*gap + 'tfidf_score' + '\n')
    for row in lemma_tfidf_ru_scores.itertuples():
        gap = space - len(row.lemma)
        f.writelines(row.lemma + ' '*gap + str(row.Tfidf_score) + '\n')

# save lemmata with their tfidf scores: Ukrainian
with open(r'lemmata_tfidf_ukr.txt', 'w', encoding='utf-8') as f:
    space = 30
    gap = space - len('lemma')
    f.writelines('lemma' + ' '*gap + 'tfidf_score' + '\n')
    for row in lemma_tfidf_ukr_scores.itertuples():
        gap = space - len(row.lemma)
        f.writelines(row.lemma + ' '*gap + str(row.Tfidf_score) + '\n')

# plot top 100 lemmata with tfidf scores
sns.barplot(x=lemma_tfidf_ru_scores.lemma[:100], y=lemma_tfidf_ru_scores.Tfidf_score[:100])
plt.xticks(rotation=90, fontsize=12)
plt.tight_layout()
plt.show(block=False)

# Russian
np.save(r'tfidf_ru_matrix.npy', tfidf_ru)
np.save(r'terms_ru.npy', vocab_ru)

# Ukrainian
np.save(r'tfidf_ukr_matrix.npy', tfidf_ukr)
np.save(r'terms_ukr.npy', vocab_ukr)

###### TOPIC MODELLING

tfidf_ru = np.load(r'tfidf_ru_matrix.npy', allow_pickle=True)
vocab_ru = np.load(r'terms_ru.npy', allow_pickle=True)

from corextopic import corextopic as ct
import os

# unsupervised
topic_model = ct.Corex(n_hidden=70, seed=42, max_iter = 400)
topic_model = topic_model.fit(
    tfidf_ukr,
    words=vocab_ukr
)

# explore topics
python explore_topics.py -m tfidf_ru_matrix.npy -t terms_ru.npy -u lemm_transcripts_list_ru.npy -s_d %cd% -d_s ru
python explore_topics.py -m tfidf_ukr_matrix.npy -t terms_ukr.npy -u lemm_transcripts_list_ukr.npy -s_d %cd% -d_s ukr

# define anchors_ru and anchors_ukr
# define anchor words: Russian
# Example:
anchors_ru = [
    
    # AGENT
    # request-agent
    ['оператор'
    , 'соединить оператор'
    , 'живой оператор'
    , 'связь оператор'
    , 'связать оператор'
    , 'можно оператор'
    , 'консультант']
]

# define anchors: Ukrainian
# Example
anchors_ukr = [
    
    # AGENT
    # request-agent
    ['звязатися оператор'
    , 'звязати оператор'
    , 'зєднатися оператор'
    , 'зєднання оператор'
    , 'зєднати оператор'
    , 'зателефонувати оператор'
    , 'звяжіти оператор'
    , 'звязка оператор'
    , 'потрібний оператор'
    , 'підключити оператор'
    , 'живий оператор'
    , 'можна звязатися оператор'
    , 'хотіти оператор'
    , 'хотіти звязатися оператор'
    , 'додзвонитися оператор'
    , 'дозвонитися оператор'
    , 'могти звязатися оператор'
    , 'подзвонити оператор'
    , 'зателефонувати оператор'
    , 'потрібно оператор'
    , 'потрібний оператор'
    , 'дзвінка оператор']
]

# filter out empty anchors
anchors_ru = [anchor for anchor in anchors_ru if anchor]
anchors_ukr = [anchor for anchor in anchors_ukr if anchor]

# filter those which are not in the model vocab: Russian
anchors_ru = [
    [a for a in topic if a in vocab_ru]
    for topic in anchors_ru
]

# filter those which are not in the model vocab: Russian
anchors_ukr = [
    [a for a in topic if a in vocab_ukr]
    for topic in anchors_ukr
]

# model with anchor words: Russian
topic_model_ru = ct.Corex(n_hidden=115, seed=42, max_iter = 100)
topic_model_ru = topic_model_ru.fit(
    tfidf_ru,
    words=vocab_ru,
    anchors=anchors_ru, # Pass the anchors in here
    anchor_strength=5 # Tell the topic_model how much it should rely on the anchors
)

# model with anchor words: Ukrainian
topic_model_ukr = ct.Corex(n_hidden=80, seed=42, max_iter = 100)
topic_model_ukr = topic_model_ukr.fit(
    tfidf_ukr,
    words=vocab_ukr,
    anchors=anchors_ukr, # Pass the anchors in here
    anchor_strength=5 # Tell the topic_model how much it should rely on the anchors
)

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_model(topic_model, dir_suffix):
    plot_dir = os.path.join(r'analysis', 'anchored_model_{}'.format(dir_suffix))
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    # save topic correlation plot
    try:
        plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
    except:
        print('Couldn\'t plot total correlation.')
    plt.xlabel('Topic', fontsize=16)
    plt.ylabel('Total Correlation (nats)', fontsize=16)
    plt.title('Number of topics: {}'.format(topic_model.tcs.shape[0]))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'topics_total_correlation.png'))
    plt.close('all')
    # append correlation numbers and kywords to a file
    with open(os.path.join(plot_dir, 'model_summary.txt'), 'a', encoding='utf-8') as f:
        f.writelines('Number of topics: {}\n'.format(topic_model.tcs.shape[0]))
        f.writelines('Total correlation is {}\n'.format(round(topic_model.tc.tolist(), 2)))
        f.writelines('------------------------------------------------------\n')
        for i, score in enumerate(topic_model.tcs.tolist()):
            f.writelines('Total correlation of topic {} is {}\n'.format(i+1, round(score, 2)))
            # write keywords to summary only if their correlation is higher than zero
            f.writelines('Corresponding keywords are {}\n'.format('\', \''.join([keyword[0] for keyword in topic_model.get_topics(n_words=10)[i] if keyword[1] > 0])))
        f.writelines('=========================================================\n')
    # loop through each topic and plot lemmata (y axis) and corresponding MI scores (x axis)
    num_lemmata=15
    topics = topic_model.get_topics(n_words=num_lemmata)
    topics_sorted = []
    for topic in topics:
        topics_sorted.append(sorted(topic, key = lambda x: x[1]))
    for j, topic in enumerate(topics_sorted):
        lemmata = [lemma[0] for lemma in topic]
        mi_scores = [score[1] for score in topic]
        fig = plt.figure(figsize=(8,4))
        # add the horizontal bar chart
        yaxis = np.arange(num_lemmata)
        try:
            ax = plt.barh(yaxis, mi_scores, align="center", color="green",tick_label=lemmata)
        except:
            print('\nCouldn\'t plot terms correlation for topic {}.'.format(j+1))
            continue
        plt.xlabel("Mutial information: lemma + topic")
        plt.title("Topic #{}".format(j+1))
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'topic_{}_terms_tc.png'.format(j+1)))
        plt.close('all')

visualize_model(topic_model, 'ru')
visualize_model(topic_model, 'ukr')

# automatically extract anchors
def compile_anchors_file(topic_model, dir_suffix):
    plot_dir = os.path.join(r'analysis', 'anchored_model_{}'.format(dir_suffix))
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    keywords = topic_model.get_topics(n_words=20)
    with open(os.path.join(plot_dir, 'anchors_file.py'), 'w', encoding='utf-8') as f:
        f.writelines('anchors = [\n\n')
        for topic_n, topic_keywords in enumerate(keywords):
            f.writelines('    ### Topic {}\n'.format(topic_n+1))
            if len(topic_keywords) == 0:
                f.writelines('    # empty \n\n')
                continue
            if topic_n == len(keywords)-1:
                for keyword_n, keyword in enumerate(topic_keywords):
                    if keyword_n == 0:
                        if len(topic_keywords) == 1:
                            f.writelines('    [\'{}\'], # {}\n\n'.format(keyword[0], round(keyword[1], 4)))
                        else:
                            f.writelines('    [\'{}\' # {}\n'.format(keyword[0], round(keyword[1], 4)))
                        if len(topic_keywords) == 1:
                            f.writelines('],\n\n')
                    elif keyword_n == len(topic_keywords)-1:
                        f.writelines('    , \'{}\'] # {}\n'.format(keyword[0], round(keyword[1], 4)))
                    else:
                        f.writelines('    , \'{}\'\n'.format(keyword[0], round(keyword[1], 4)))
            else:
                for keyword_n, keyword in enumerate(topic_keywords):
                    if keyword_n == 0:
                        if len(topic_keywords) == 1:
                            f.writelines('    [\'{}\'], # {}\n\n'.format(keyword[0], round(keyword[1], 4)))
                        else:
                            f.writelines('    [\'{}\' # {}\n'.format(keyword[0], round(keyword[1], 4)))
                    elif keyword_n == len(topic_keywords)-1:
                        f.writelines('    , \'{}\'], # {}\n\n'.format(keyword[0], round(keyword[1], 4)))
                    else:
                        f.writelines('    , \'{}\' # {}\n'.format(keyword[0], round(keyword[1], 4)))
        f.writelines(']\n')

compile_anchors_file(topic_model, 'ru')
compile_anchors_file(topic_model, 'ukr')

# extract tag list from an anchors file
def extract_tag_list(anchors_file):
    with open(anchors_file, 'r', encoding='utf-8') as f:
        file_contents = f.readlines()
        tag_list = []
        for i, line in enumerate(file_contents):
            if re.search('###', line):
                tag_list.append(line.split()[1])
    return tag_list

tag_list_ru = extract_tag_list(r'anchors_ru\full_tags_for_anchors_ru.py')
tag_list_ukr = extract_tag_list(r'anchors_ru\full_tags_for_anchors_ukr.py')

# extract example transcripts for each topic
def extract_transcripts(topic_model, transcript):
    plot_dir = os.path.join(r'analysis', 'anchored_model')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    top_docs = topic_model.get_top_docs(n_docs=1000) # default n_docs=10 per topic
    keywords = topic_model.get_topics(n_words=10)
    num_examples = 25 # number of examples to write to file
    docs_num = []
    with open(os.path.join(plot_dir, 'transcript_examples.txt'), 'a', encoding='utf-8') as f1, \
        open(os.path.join(plot_dir, 'topic_table.txt'), 'a', encoding='utf-8') as f2:
        for topic_n, topic_docs in enumerate(top_docs):
            # exclude docs with log_probability score lower than zero
            topic_docs = [doc for doc in topic_docs if abs(doc[1]) < 1e-2]
            try:
                docs,probs = zip(*topic_docs)
                docs = list(docs)
                docs_num.append(len(docs))
                keyword_str = '\', \''.join([keyword[0] for keyword in keywords[topic_n] if keyword[1] > 0])
                f1.writelines('\n\n===========================================\n')
                f1.writelines('Topic {} with keywords: {}\n\n'.format(topic_n+1, keyword_str))
                f1.writelines('Number of transcripts for the topic: {}\n\n'.format(len(docs)))
                f1.writelines('========= Example transcripts =============\n\n')
                for x in docs[:num_examples]:
                    f1.writelines(transcript[x] + '\n')
                    f1.writelines('-------------------------------------------\n')
                f2.writelines('Topic {} has {} transcripts.\n'.format(topic_n+1, len(docs)))
            except:
                print('All docs filted out for topic {}.'.format(topic_n+1))
                keyword_str = '\', \''.join([keyword[0] for keyword in keywords[topic_n] if keyword[1] > 0])
                f1.writelines('\n\n===========================================\n')
                f1.writelines('Topic {} with keywords: {}\n\n'.format(topic_n+1, keyword_str))
                f1.writelines('Number of transcripts for the topic: 0')
                f1.writelines('======= No example transcripts =============\n\n')
        f1.writelines('===========================================\n\n')
        f1.writelines('With {} topics approximately {}% of transcripts were labeled.'.format(len(top_docs), \
            round(np.sum(docs_num)/topic_model.p_y_given_x.shape[0]*100, 2)))
        f2.writelines('===========================================\n\n')
        f2.writelines('With {} topics approximately {}% of transcripts were labeled.'.format(len(top_docs), \
            round(np.sum(docs_num)/topic_model.p_y_given_x.shape[0]*100, 2)))

extract_transcripts(topic_model, transcripts_list_ru)
extract_transcripts(topic_model, transcripts_list_ukr)

# SEPARATE FUNCTION CALLS
# list identified topics and corresponding keywords
for i, keywords in enumerate(topic_model.get_topics(n_words=15)):
    keywords = [keyword[0] for keyword in keywords if keyword[1] > 0]
    print("Topic #{}: {}".format(i+1, ", ".join(keywords)))

topic_model_ru.tc
topic_model_ukr.tc
# 15.27 for Russian
# 12.507142202311535 for Ukrainian

# for single topics
topic_model.tcs

# topic correlation plot
#plt.figure(figsize=(10,5))
plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16)
plt.show(block=False)

# loop through each topic and plot lemmata (y axis) and corresponding MI scores (x axis)
num_lemmata=15
topics = topic_model.get_topics(n_words=num_lemmata)
topics_sorted = []
for topic in topics:
    topics_sorted.append(sorted(topic, key = lambda x: x[1]))

for i, topic in enumerate(topics_sorted):
    lemmata = [lemma[0] for lemma in topic]
    mi_scores = [score[1] for score in topic]
    fig = plt.figure(figsize=(8,4))
    # add the horizontal bar chart
    yaxis = np.arange(num_lemmata)
    ax = plt.barh(yaxis, mi_scores, align="center", color="green",tick_label=lemmata)
    plt.xlabel("Mutial information: lemma + topic")
    plt.title("Topic #{}".format(i+1))
    plt.show(block=False)

# get most probable documents for each topic
top_docs = topic_model.get_top_docs(n_docs=3)
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    docs = list(docs)
    docs_str = [str(doc) for doc in docs]
    print('\n===============================')
    print('Topic ' + str(topic_n+1))
    print('\n===============================')
    for index in docs:
        print(transcript_ru[index])
        print('\n        -----        ')

# decompose correlation across documents (useful to assess the optimal number of topics)
cor_scores = [max(cor_score) for cor_score in topic_model.log_z]
plt.figure(figsize=(10,5))
plt.bar(range(topic_model.log_z.shape[0]), cor_scores, color='#4e79a7', width=0.5)
plt.xlabel('Documents', fontsize=16)
plt.ylabel('Topic Correlation', fontsize=16)
plt.show(block=False)

########### TOPICS AUTO TAGGING

from collections import Counter

# ADD TOP LEMMATA AS TAGGING HELP to the data frame
def add_lemma_transcripts(target_df, lemm_transcripts_list, morph):
    noun_pos_tags = ['NOUN']
    pred_pos_tags = ['INFN', 'PRED', 'ADVB', 'ADJF']
    empty_token = 'NA'
    first_pred = []
    second_pred = []
    first_noun = []
    second_noun = []
    for i, transcript in enumerate(tqdm(lemm_transcripts_list)):
        preds_transcript = []
        nouns_transcript = []
        for lemma in transcript.split():
            if morph.parse(lemma)[0].tag.POS in pred_pos_tags and len(preds_transcript) < 2:
                preds_transcript.append(lemma)
            if morph.parse(lemma)[0].tag.POS in noun_pos_tags and len(nouns_transcript) < 2:
                nouns_transcript.append(lemma)
        num_empty_preds = 2-len(preds_transcript)
        for i in range(num_empty_preds):
            preds_transcript.append(empty_token)
        num_empty_nouns = 2-len(nouns_transcript)
        for i in range(num_empty_nouns):
            nouns_transcript.append(empty_token)
        first_pred.append(preds_transcript[0])
        second_pred.append(preds_transcript[1])
        first_noun.append(nouns_transcript[0])
        second_noun.append(nouns_transcript[1])
    target_df['first_pred'] = first_pred
    target_df['second_pred'] =  second_pred
    target_df['first_noun'] = first_noun
    target_df['second_noun'] =  second_noun
    return target_df

df_ru = add_lemma_transcripts(df_ru, lemm_transcripts_list_ru, morph_ru)
df_ukr = add_lemma_transcripts(df_ukr, lemm_transcripts_list_ukr, morph_ukr)

# assign each document a topic with highest probability or no topic
def label_transcripts_model_prob(topic_model, tag_list):
    prob_threshold=0.9
    sem_tag_list = []
    for doc_index, probabilities in enumerate(topic_model.p_y_given_x):
        probabilities = probabilities.tolist()
        if max(probabilities) >= prob_threshold:
            topic_index = int(probabilities.index(max(probabilities)))
            topic = tag_list[topic_index]
            #topic = 'Topic ' + str(topic_index)
            sem_tag_list.append(topic)
        else:
            #sem_tag_list.append('vague-vague')
            sem_tag_list.append('no-match')
    return sem_tag_list

sem_tag_list_ru = label_transcripts_model_prob(topic_model_ru, tag_list_ru)
sem_tag_list_ukr = label_transcripts_model_prob(topic_model_ukr, tag_list_ukr)

# look for at least 2 keyword matches with each key_gram
def label_transcripts_key_gram(lemm_transcripts_clean, anchors):
    min_num_matches = 2
    sem_tag_list = []
    for index, transcript in enumerate(tqdm(lemm_transcripts_clean, leave=False)):
        sem_tag_set = 0
        for key in anchors:
            matches = 0
            for key_gram in anchors[key]:
                if sem_tag_set:
                    break
                for word in key_gram.split():
                    if re.match(word, transcript): #if re.match(key_gram, transcript):
                        matches = matches + 1
                        # print('\nKeyword match')
                        # breakpoint()
                        if matches >= min_num_matches:
                            # print('\nTwo matches reached.')
                            # breakpoint()
                            sem_tag_list.append(key)
                            sem_tag_set = 1
                            break
        if not sem_tag_set:
            sem_tag_list.append('vague-vague')
    return sem_tag_list

# combine tag list and anchors into a dict
anchors_dict_ru = dict(zip(tag_list_ru, anchors_ru))
anchors_dict_ukr = dict(zip(tag_list_ukr, anchors_ukr))

sem_tag_list_ru = label_transcripts_key_gram(lemm_transcripts_list_ru, anchors_dict_ru)
sem_tag_list_ukr = label_transcripts_key_gram(lemm_transcripts_list_ukr, anchors_dict_ukr)

# get percentage of transcripts without prediction
round(df_ru.sem_tag[df_ru.sem_tag == 'vague-vague'].count()/df_ru.shape[0]*100, 2) # about 42%
round(df_ukr.sem_tag[df_ukr.sem_tag == 'vague-vague'].count()/df_ukr.shape[0]*100, 2) # about 40%

# decide based on tag frequency whether to take model tag or key_gram_match
def label_transcripts(target_df, topic_model, tag_list, lemm_transcripts_list, anchors_dict):
    model_sem_tags = label_transcripts_model_prob(topic_model, tag_list)
    key_gram_sem_tags = label_transcripts_key_gram(lemm_transcripts_list, anchors_dict)
    tag_counts = dict(Counter(model_sem_tags))
    merged_sem_tags = []
    for index, tag in enumerate(model_sem_tags):
        # replace model_tag with key_gram tag if number of corresponding transcripts lower than 2%
        if round(tag_counts[tag]/target_df.shape[0]*100) < 3 or tag == 'no-match':
            merged_sem_tags.append(key_gram_sem_tags[index])
        else:
            merged_sem_tags.append(model_sem_tags[index])
    target_df['sem_tag'] = np.nan
    target_df['sem_tag'] = merged_sem_tags
    counts = Counter(merged_sem_tags)
    return target_df, counts

df_ru, counts_ru = label_transcripts(df_ru, topic_model_ru, tag_list_ru, lemm_transcripts_list_ru, anchors_dict_ru)
df_ukr, counts_ukr = label_transcripts(df_ukr, topic_model_ukr, tag_list_ukr, lemm_transcripts_list_ukr, anchors_dict_ukr)

# SAVE tagged data to tables
df_counts_ru = pd.DataFrame(list(counts_ru.items()),columns = ['sem_tag','counts'])
df_counts_ukr = pd.DataFrame(list(counts_ukr.items()),columns = ['sem_tag','counts'])

# add tag_list to the tagging sheet
tag_list_complete = extract_tag_list(r'anchors_ru\full_tags.py')
# compile a project phase column
phase_list = []
for tag in tag_list_complete:
    if tag in tag_list_phase_1:
        phase_list.append('phase_1')
    else:
        phase_list.append('NA')
tag_list = pd.DataFrame(list(zip(tag_list_complete, phase_list)), columns = ['tag', 'project_phase'])

writer = pd.ExcelWriter(r'tagging_sheet.xlsx', engine = 'openpyxl') # xlsxwriter
df_ru.to_excel(writer, sheet_name = 'russian_tagging_sheet', index = False)
tag_list.to_excel(writer, sheet_name = 'tag_list', index = False)
writer.save()
writer.close()

writer = pd.ExcelWriter(r'Russian_transcripts.xlsx', engine = 'openpyxl') # xlsxwriter
df_ru.to_excel(writer, sheet_name = 'tagging_sheet', index = False)
writer.save()
writer.close()

writer = pd.ExcelWriter(r'Ukrainian_transcripts.xlsx', engine = 'openpyxl') # xlsxwriter
df_ukr.to_excel(writer, sheet_name = 'tagging_sheet', index = False)
writer.save()
writer.close()