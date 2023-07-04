import datetime
import math
import string
import gensim
import nltk
import pandas as pd
import preprocess_data as preprocess_data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import re
from gensim import corpora
from nltk.corpus import stopwords
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


#data preprocessing : correct the wrong sentences in english, remove the stop words and punctuation, tokenize the sentences after getting the from the document, convert into lower cased
#text blob corrects english sentences
#a service for data preprocessing and then service for dynamic segmentation and topic modeling

# def data_preprocessing(data):
#     data = data.lower()
#     data = re.sub(r'\d+', '', data)
#     data = data.translate(str.maketrans('', '', string.punctuation))
#     data = data.strip()
#     data = data.split()
#     data = [word for word in data if word not in stopwords.words('english')]
#     data = [word for word in data if len(word) > 2]


#Todo: change the dummy data with real data that will get preprocessed
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


def retrieve_documents_between_time_period(data, x1, x2):
    documents = []
    x1_datetime = datetime.datetime.strptime(x1, '%Y-%m-%d %H:%M:%S')
    x2_datetime = datetime.datetime.strptime(x2, '%Y-%m-%d %H:%M:%S')

    for item in data:
        timestamp = datetime.datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S')
        if x1_datetime <= timestamp <= x2_datetime:
            documents.append(item[1])

    return documents


#Todo: to be checked if it works correctly or not
def apply_LDA(window,num_topics):
    return preprocess_data.apply_lda_model([data[1] for data in window], num_topics)

#Todo: to be checked if it works correctly or not
def calculate_contingency_matrix(data1, data2, num_topics):
    topics_words_seg1 = apply_LDA(data1, num_topics)
    topics_words_seg2 = apply_LDA(data2, num_topics)
    # print(topics_words_seg1)
    # print(topics_words_seg2)
    contingency_matrix = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            for k in range(len(topics_words_seg1[i])):
                if(topics_words_seg1[i][k] in topics_words_seg2[j]):
                    contingency_matrix[i][j] += 1
    return contingency_matrix
#Todo: to be checked if it works correctly or not
def calculate_objective_function(contingency_matrix):
    r, c = contingency_matrix.shape
    u_C = np.full(c, 1 / c)
    u_R = np.full(r, 1 / r)
    F=0.0
    for i in range(r):
        for j in range(c):
            if(np.sum(contingency_matrix[i]) == 0):
                P_ri=0
            else:
                P_ri = contingency_matrix[i][j] / np.sum(contingency_matrix[i])
            if(np.sum(contingency_matrix[:, j]) == 0):
                P_cj = 0
            else:
                P_cj = contingency_matrix[i][j] / np.sum(contingency_matrix[:, j])
            if P_ri != 0 and P_cj != 0:
                kl_divergence_r = np.sum(P_ri * np.log(u_C/P_ri))
                kl_divergence_c = np.sum(P_cj * np.log(u_R/P_cj))
                F += (1/r)*kl_divergence_r + (1/c)*kl_divergence_c
            else:
                F += 0
    return F


#Todo: should be tested if it works correctly or not
def perform_segmentation_with_timestamps(data, x, y):
    segments = []
    T = [datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') for d in data]
    t_t = T[-1]
    W1Start = T[0]
    W1Size = x
    F = float('-inf')
    while W1Start + datetime.timedelta(days=W1Size.days + x.days) <= t_t and W1Size <= y:
        W2Start = W1Start + datetime.timedelta(days=W1Size.days + 1)  # Adding 1 day
        W2Size = x
        Conversion = False
        while W2Start + datetime.timedelta(days=W2Size.days) <= t_t and W2Size <= y:
            W1 = [(datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S'), d[1]) for d in data if datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') >= W1Start and
                  datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') < W1Start + datetime.timedelta(days=W1Size.days)]
            W2 = [(datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S'), d[1]) for d in data if datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') >= W2Start and
                  datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') < W2Start + datetime.timedelta(days=W2Size.days)]
            window1 = apply_LDA(W1,5)
            window2 = apply_LDA(W2,5)
            contingency_matrix=calculate_contingency_matrix(W1,W2,5)
            print(contingency_matrix)
            print(window1)
            print(window2)
            F0 = calculate_objective_function(contingency_matrix)
            print(F0)
            if F0 > F or W1Size.days == y.days or W2Size.days == y.days:
                segment1_start = W1[0][0]
                segment1_end = W1[-1][0]
                segment2_start = W2[0][0]
                segment2_end = W2[-1][0]
                segments.append((segment1_start, segment1_end, window1))
                segments.append((segment2_start, segment2_end, window2))
                W1Start = W2Start + datetime.timedelta(days=W2Size.days + 1)  # Adding 1 day
                W1Size = x
                Conversion = True
                print('there were a conversion')
                break
            F = F0
            W2Size += x
        if not Conversion:
            print('there were no conversion')
            W1Size += x

    leftover_data = [(datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S'), d[1]) for d in data if datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') >= W1Start and
                     datetime.datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') <= t_t]
    segment_start = leftover_data[0][0]
    segment_end = leftover_data[-1][0]
    leftover_data = apply_LDA(leftover_data,5)
    segments.append((segment_start, segment_end, leftover_data))
    return segments


x = datetime.timedelta(days=4)  # Minimum window size
y = datetime.timedelta(days=16)  # Maximum window size
segments = perform_segmentation_with_timestamps(data, x, y)
print(len(segments))
i=0
for segment in segments:
    print(f'here is the segment {i} {segment}')
    i+=1

