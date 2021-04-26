from bertopic import BERTopic

# BERTopic is using sentence-transformers to create embeddings for the documents you pass it
# language="multilingual" to select a model that supports over 50 languages
# BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens") to choose other models
# list of pretrained embeddings: https://www.sbert.net/docs/pretrained_models.html
# nr_topics="auto" for topic auto-reduction
bert_model = BERTopic(language="Russian", nr_topics=70, calculate_probabilities=True)
# takes around 25 minutes to train on a 5.5k vocabulary
topics, probabilities = bert_model.fit_transform(docs)
bert_topics, bert_probabilities = bert_model.fit_transform(transcripts_clean_ru)



# get topic counts
bert_model.get_topic_freq().head()
# check specific topic
bert_model.get_topic(39)

bert_model.save("my_model")
bert_model = BERTopic.load("my_model")

# check probability distribution (doesn't work)
bert_model.visualize_distribution(bert_probabilities)

# reduce topics of a trained model (doesn't work)
new_bert_topics, new_bert_probs = bert_model.reduce_topics(transcripts_clean_ru, bert_topics, bert_probabilities, nr_topics=60)

# clusters
bert_figure = bert_model.visualize_topics()
bert_figure.show()

# Create topics over time
model = BERTopic(verbose=True)
topics, _ = model.fit_transform(tweets)
topics_over_time = model.topics_over_time(tweets, topics, timestamps)

model.visualize_topics_over_time(topics_over_time, topcs=[9, 10, 72, 83, 87, 91])