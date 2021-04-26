''' Script which fits a series of topic models and provides corresponding summaries,
    metrics, and visualizations for each model. The output can be used to define anchors
    for a semi-supervised topic model (see https://github.com/gregversteeg/corex_topic)

    Author Eugen Klein, February 2021 '''

import os
import argparse
from tqdm import tqdm

import numpy as np
from corextopic import corextopic as ct

import matplotlib.pyplot as plt
import seaborn as sns


def extract_model_info(tfidf_matrix, terms, transcripts, num_topics, num_keywords, save_dir, dir_suffix):

    main_dir = os.path.join(save_dir,'explore_topics_{}'.format(dir_suffix))
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    
    # skip models with less than 10 topics and add 10 topics during each modelling step
    topic_numbers = [x for x in range(0, num_topics, 10)][1:]
    total_correlation = {}
    transcript_indx = []

    # modelling loop
    for i, topic_number in enumerate(tqdm(topic_numbers)):

        # n_hidden is the number of topics the topic_model targets for
        topic_model = ct.Corex(n_hidden=topic_number, seed=42, max_iter=400)
        topic_model = topic_model.fit(tfidf_matrix, words=terms)

        # extract topics with corresponding keywords
        topics = topic_model.get_topics(n_words=num_keywords)

        # extract top docs assigned to topics
        top_docs = topic_model.get_top_docs(n_docs=1000)

        plot_dir = os.path.join(main_dir, '{}_topics_model'.format(topic_number))
        os.mkdir(plot_dir)

        # save total correlation plot
        try:
            plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color='#4e79a7', width=0.5)
        except:
            print('Couldn\'t plot total correlation.')
            continue
        plt.xlabel('Topic', fontsize=16)
        plt.ylabel('Total Correlation (nats)', fontsize=16)
        plt.title('Number of topics: {}'.format(topic_number))
        plt.tight_layout()
        plt.savefig(os.path.join(main_dir, '{}_topics_total_correlation.png'.format(topic_number)))
        plt.close('all')

        # append model correlation numbers and keywords to a summary file
        with open(os.path.join(main_dir, 'correlation_keywords_summary.txt'), 'a', encoding='utf-8') as f:
            f.writelines('Number of topics: {}\n'.format(len(topic_model.tcs.tolist())))
            f.writelines('Total correlation is {}\n'.format(round(topic_model.tc.tolist(), 2)))
            f.writelines('------------------------------------------------------\n')
            for i, score in enumerate(topic_model.tcs.tolist()):
                f.writelines('Total correlation of topic {} is {}\n'.format(i+1, round(score, 2)))
                # write keywords to summary only if their correlation is higher than zero
                f.writelines('Corresponding keywords are {}\n'.format('\', \''.join([keyword[0] for keyword in topics[i] if keyword[1] > 0])))
            f.writelines('=========================================================\n')
        
        # write total correlation scores to a dict
        total_correlation[len(topic_model.tcs.tolist())] = round(topic_model.tc.tolist(), 2)

        # loop through each topic and plot keywords/lemmata (y axis) and corresponding MI scores (x axis)
        topics_sorted = []
        for topic in topics:
            topics_sorted.append(sorted(topic, key = lambda x: x[1]))

        for j, topic in enumerate(topics_sorted):
            lemmata = [lemma[0] for lemma in topic]
            mi_scores = [score[1] for score in topic]
            fig = plt.figure(figsize=(8,4))
            # add the horizontal bar chart
            yaxis = np.arange(num_keywords)
            try:
                ax = plt.barh(yaxis, mi_scores, align="center", color="green", tick_label=lemmata)
            except:
                print('\nCouldn\'t plot terms correlation for topic {}.'.format(j+1))
                continue
            plt.xlabel("Mutial information: lemma + topic")
            plt.title("Topic #{}".format(j+1))
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'topic_{}_terms_tc.png'.format(j+1)))
            plt.close('all')

        # extract transcript examples for each topic and compile percentage per topic table
        docs_num = []
        num_examples = 25 # number of transcript examples to write to file
        with open(os.path.join(plot_dir, '{}_topics_transcript_examples.txt'.format(topic_number)), 'a', encoding='utf-8') as f1, \
            open(os.path.join(plot_dir, '{}_topics_percentage_table.txt'.format(topic_number)), 'a', encoding='utf-8') as f2:
            for topic_n, topic_docs in enumerate(top_docs):
                # exclude docs with log_probability score lower than zero
                topic_docs = [doc for doc in topic_docs if abs(doc[1]) < 1e-2]
                try:
                    docs,probs = zip(*topic_docs)
                    docs = list(docs)
                    docs_num.append(len(docs))
                    keyword_str = ', '.join([keyword[0] for keyword in topics[topic_n] if keyword[1] > 0])
                    f1.writelines('\n\n===========================================\n')
                    f1.writelines('Topic {} with keywords: {}\n\n'.format(topic_n+1, keyword_str))
                    f1.writelines('Number of transcripts for the topic: {}\n\n'.format(len(docs)))
                    f1.writelines('========= Example transcripts =============\n\n')
                    for index in docs[:num_examples]:
                        f1.writelines(transcripts[index] + '\n')
                        f1.writelines('-------------------------------------------\n')
                    f2.writelines('Topic {} has {} transcripts.\n'.format(topic_n+1, len(docs)))
                except:
                    print('All docs filted out for topic {}.'.format(topic_n+1))
                    keyword_str = '\', \''.join([keyword[0] for keyword in topics[topic_n] if keyword[1] > 0])
                    f1.writelines('\n\n===========================================\n')
                    f1.writelines('Topic {} with keywords: {}\n\n'.format(topic_n+1, keyword_str))
                    f1.writelines('Number of transcripts for the topic: 0\n\n')
                    f1.writelines('======= No example transcripts =============\n\n')
            f1.writelines('===========================================\n\n')
            f1.writelines('With {} topics approximately {}% of transcripts were labeled.'.format(len(top_docs), \
                round(np.sum(docs_num)/topic_model.p_y_given_x.shape[0]*100, 2)))
            f2.writelines('===========================================\n\n')
            f2.writelines('With {} topics approximately {}% of transcripts were labeled.'.format(len(top_docs), \
                round(np.sum(docs_num)/topic_model.p_y_given_x.shape[0]*100, 2)))

        # compile the anchors file for each model
        with open(os.path.join(main_dir, 'anchors_file_{}_topics.py'.format(topic_number)), 'w', encoding='utf-8') as f:
            f.writelines('anchors = [\n\n')
            for topic_n, topic_keywords in enumerate(topics):
                f.writelines('    ### Topic {}\n'.format(topic_n+1))
                if len(topic_keywords) == 0:
                    f.writelines('    # empty \n\n')
                    continue
                if topic_n == len(topics)-1:
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

    # add info about the model with highst total correlation to the summary
    max_correlation = max(total_correlation.values())
    optimal_topic_num = [topic_num for topic_num, correlation in total_correlation.items() if correlation == max_correlation]

    with open(os.path.join(main_dir, 'correlation_keywords_summary.txt'), 'a', encoding='utf-8') as f:
        f.writelines('With {} topics max total correlation is {}.\n'.format(optimal_topic_num[0], max_correlation))
        f.writelines('=========================================================\n')

def main(args):
    
    tfidf_matrix_file = args.matrix
    terms_file = args.terms
    transcripts_file = args.utterances
    num_topics = args.number_topics
    num_keywords = args.number_keywords
    save_dir = args.save_dir
    dir_suffix = args.dir_suffix

    tfidf_matrix = np.load(tfidf_matrix_file, allow_pickle=True)
    terms = np.load(terms_file, allow_pickle=True)
    tfidf_matrix = tfidf_matrix.tolist()
    terms = terms.tolist()
    transcripts = np.load(transcripts_file, allow_pickle=True)
    transcripts = transcripts.tolist()
    extract_model_info(tfidf_matrix, terms, transcripts, num_topics, num_keywords, save_dir, dir_suffix)

if __name__ == "__main__":

    ''' Visual exploration of Tfidf matrix with respect to underlying topics. '''

    parser = argparse.ArgumentParser(description='Explore topic modelling \
        for a Tfidf matrix.')

    parser.add_argument('-m', '--matrix', required=True, \
        help="Tfidf matrix.")
    parser.add_argument('-t', '--terms', required=True, \
        help="List of terms from the Tfidf matrix.")
    parser.add_argument('-u', '--utterances', required=True, \
        help="List of lemmatized utterances/transcripts.")
    parser.add_argument('-n_t', '--number_topics', default=70, \
        help="Number of max topics.")
    parser.add_argument('-n_k', '--number_keywords', default=20, \
        help="Number of kewords to extract per topic.")
    parser.add_argument('-s_d', '--save_dir', required=True, \
        help="Directory to save exploration plots.")
    parser.add_argument('-d_s', '--dir_suffix', required=True, \
        help="Directory to save exploration plots.")

    args = parser.parse_args()

    main(args)