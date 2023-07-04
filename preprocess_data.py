from matplotlib import pyplot as plt
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

data = [
    ['2022-01-01 09:00:00', 'The latest fashion trends for the upcoming season are vibrant colors and bold patterns.'],
    ['2022-01-02 12:30:00', 'Celebrity gossip magazines are reporting on a high-profile breakup in Hollywood.'],
    ['2022-01-03 15:45:00', 'Food bloggers are sharing their favorite recipes for healthy smoothies and salads.'],
    ['2022-01-04 08:15:00', 'Fitness influencers are promoting a new workout routine for achieving toned abs.'],
    ['2022-01-05 14:00:00', 'Technology websites are reviewing the latest smartphone models and their features.'],
    ['2022-01-06 10:45:00', 'Travel bloggers are documenting their adventures in exotic destinations around the world.'],
    ['2022-01-07 16:30:00', 'The art exhibition showcases a diverse collection of contemporary artwork.'],
    ['2022-01-08 11:00:00', 'Political analysts are discussing the impact of recent election results on the economy.'],
    ['2022-01-09 13:45:00', 'Home improvement experts are sharing tips for renovating and decorating on a budget.'],
    ['2022-01-10 09:30:00', 'The new movie release is generating buzz among film enthusiasts and critics.'],
    ['2022-01-11 14:20:00', 'Health experts are highlighting the importance of regular exercise for overall well-being.'],
    ['2022-01-12 17:15:00', 'The fashion industry is anticipating the launch of new designer collections for the season.'],
    ['2022-01-13 10:10:00', 'Travel agencies are offering discounted packages for summer vacation destinations.'],
    ['2022-01-14 12:50:00', 'The tech conference will feature presentations on emerging technologies and innovation.'],
    ['2022-01-15 15:35:00', 'Cooking enthusiasts are experimenting with unique flavor combinations and techniques.'],
    ['2022-01-16 09:20:00', 'Financial advisors are providing tips for saving and investing for retirement.'],
    ['2022-01-17 13:10:00', 'Artists are showcasing their work at local galleries and exhibitions.'],
    ['2022-01-18 16:40:00', 'The sports event attracted a large crowd of passionate fans from around the world.'],
    ['2022-01-19 11:30:00', 'Technology companies are unveiling their latest products and innovations at the expo.'],
    ['2022-01-20 14:15:00', 'Fitness enthusiasts are participating in a marathon to support a charitable cause.'],
    ['2022-01-21 17:00:00', 'Travel vloggers are sharing their experiences and recommendations for popular destinations.'],
    ['2022-01-22 10:50:00', 'The music festival will feature performances by renowned artists and bands.'],
    ['2022-01-23 12:25:00', 'Environmental organizations are raising awareness about the importance of conservation.'],
    ['2022-01-24 15:55:00', 'Fashion designers are incorporating sustainable practices into their collections.'],
    ['2022-01-25 09:40:00', 'Technology enthusiasts are eagerly awaiting the release of the latest gadgets.'],
    ['2022-01-26 14:30:00', 'Food critics are reviewing new restaurants and sharing their culinary experiences.'],
    ['2022-01-27 11:15:00', 'The business conference will feature talks by industry leaders and experts.'],
    ['2022-01-28 13:55:00', 'Art enthusiasts are exploring galleries and museums to discover new artistic creations.'],
    ['2022-01-29 16:50:00', 'The book club members are discussing their thoughts on the latest best-selling novel.'],
    ['2022-01-30 10:35:00', 'Fashion bloggers are showcasing their style choices and offering fashion advice.'],
    ['2022-01-31 12:40:00', 'The film festival will screen a diverse selection of international movies and documentaries.'],
    ['2022-02-01 09:00:00', 'The automotive industry is introducing electric vehicles with advanced features.'],
    ['2022-02-02 12:30:00', 'Business leaders are attending a networking event to foster collaboration and growth.'],
    ['2022-02-03 15:45:00','Artificial intelligence is revolutionizing various sectors with automation and machine learning.'],
    ['2022-02-04 08:15:00', 'Gaming enthusiasts are excited about the release of a highly anticipated video game.'],
    ['2022-02-05 14:00:00', 'Educational institutions are adapting to online learning and digital classrooms.'],
    ['2022-02-06 10:45:00', 'Investors are exploring new opportunities in cryptocurrency and blockchain technology.'],
    ['2022-02-07 16:30:00', 'Fashion designers are experimenting with sustainable and eco-friendly materials.'],
    ['2022-02-08 11:00:00', 'Healthcare professionals are developing innovative treatments and therapies.'],
    ['2022-02-09 13:45:00', 'Travel agencies are offering vacation packages for remote work and digital nomads.'],
    ['2022-02-10 09:30:00','Sports enthusiasts are following major tournaments and championships in various disciplines.'],
    ['2022-02-11 14:20:00','Food delivery services are gaining popularity with convenient online ordering and fast delivery.'],
    ['2022-02-12 17:15:00','Environmental activists are advocating for sustainable practices and conservation efforts.'],
    ['2022-02-13 10:10:00', 'The entertainment industry is producing captivating content for streaming platforms.'],
    ['2022-02-14 12:50:00', 'The stock market is experiencing fluctuations due to global economic conditions.'],
    ['2022-02-15 15:35:00', 'Art collectors are investing in unique and valuable pieces of artwork.'],
    ['2022-02-16 09:20:00', 'Technology startups are disrupting traditional industries with innovative solutions.'],
    ['2022-02-17 13:10:00', 'Book lovers are exploring new genres and authors for their reading lists.'],
    ['2022-02-18 16:40:00', 'The hospitality industry is adapting to changing travel trends and preferences.'],
    ['2022-02-19 11:30:00', 'Financial institutions are introducing digital banking services and mobile apps.'],
    ['2022-02-20 14:15:00','Artificial reality and virtual reality technologies are transforming the gaming and entertainment experiences.'],
    ['2022-02-21 17:00:00','Social media influencers are partnering with brands for sponsored content and collaborations.'],
    ['2022-02-22 10:50:00', 'The music industry is showcasing emerging talent through music streaming platforms.'],
    ['2022-02-23 12:25:00','Online marketplaces are providing opportunities for small businesses and independent sellers.'],
    ['2022-02-24 15:55:00', 'Fashionistas are attending fashion shows to discover the latest trends and collections.'],
    ['2022-02-25 09:40:00','Technology companies are developing smart home devices for increased convenience and automation.'],
    ['2022-02-26 14:30:00','Fitness apps and wearable devices are helping individuals track their health and fitness goals.'],
    ['2022-02-27 11:15:00','The automotive industry is investing in autonomous vehicle technology for safer and efficient transportation.'],
    ['2022-02-28 13:55:00','The film industry is recognizing outstanding achievements in cinema through award ceremonies.'],
    ['2022-03-01 09:00:00','The gaming industry is expanding its reach with cross-platform gaming and esports competitions.'],
    ['2022-03-02 12:30:00', 'Entrepreneurs are launching startups to address social and environmental challenges.'],
    ['2022-03-03 15:45:00', 'Online learning platforms are providing access to education for individuals worldwide.'],
    ['2022-03-04 08:15:00', 'Social movements are advocating for equality, justice, and positive societal change.'],
    ['2022-03-05 14:00:00','Digital marketing strategies are evolving with the rise of influencer marketing and personalized advertising.'],
    ['2022-03-06 10:45:00','Fitness classes and workout apps are offering interactive and engaging exercise experiences.'],
    ['2022-03-07 16:30:00','The fashion industry is embracing body positivity and inclusivity in its campaigns and designs.'],
    ['2022-03-08 11:00:00','Health and wellness retreats are gaining popularity as people prioritize self-care and relaxation.'],
    ['2022-03-09 13:45:00','Adventure travel and outdoor activities are attracting thrill-seekers and nature enthusiasts.'],
    ['2022-03-10 09:30:00','E-commerce platforms are reshaping the retail industry with convenient online shopping experiences.'],
    ['2022-03-11 14:20:00', 'Educational technology tools are enhancing classroom learning and student engagement.'],
    ['2022-03-12 17:15:00', 'Digital nomads are embracing remote work and traveling while maintaining their careers.'],
    ['2022-03-13 10:10:00','The entertainment industry is producing diverse and inclusive content for global audiences.'],
    ['2022-03-14 12:50:00', 'Cryptocurrencies are gaining acceptance as a form of payment and investment.'],
    ['2022-03-15 15:35:00', 'Artificial intelligence-powered chatbots are improving customer service and support.'],
    ['2022-03-16 09:20:00','Renewable energy sources such as solar and wind power are becoming more accessible and affordable.'],
]
def preprocess(data):
    # Keep only nouns, adjectives, verbs and adverbs and remove other not needed tags
    removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']
    # Get tokens
    tokens = []
    # Load spacy model for processing text
    nlp = en_core_web_md.load()
    # Tokenize data
    for text in nlp.pipe(data):
        token = [token.lemma_.lower() for token in text if
                 token.pos_ not in removal and not token.is_stop and token.is_alpha]
        tokens.append(token)

    return tokens
def apply_lda_model(data, num_topics):
    tokens = preprocess(data)
    topics_keywords = []
    # Create dictionary and corpus for LDA model
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]
    # Train LDA model
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=num_topics, workers=4,
                             passes=10)
    # Get topics and keywords
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)
    for idx, topic in topics:
        keywords = [word for word, _ in topic]
        topics_keywords.append(keywords)

    return topics_keywords
def get_topics_number(dictionary, corpus, tokens):
    topics = []
    score = []
    for i in range(1,20,1):
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers = 4, passes=10, random_state=100)
        cm = CoherenceModel(model=lda_model, texts = tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
        topics.append(i)
        score.append(cm.get_coherence())
    plt.plot(topics, score)
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.show()

# Test the methods defined above
data = [row[1] for row in data]
tokens = preprocess(data)
print(tokens)
# topics, dictionary, corpus = apply_lda_model(data, 5)
# print(topics)


