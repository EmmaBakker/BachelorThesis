import pandas as pd

# MERGE MECO DF WITH LSTM RESULTS

en_merged = pd.read_csv("en_merged.csv")
tf_output = pd.read_csv("EN_tf_test_results.csv")

# remove the end of sentence tag for the target word
tf_output = tf_output[tf_output['actual_word'] != '</s>'].reset_index()
tf_output.rename(columns={"correct": "tf_correct", "previous_word": "tf_previous_word",
                           "predicted_word": "tf_predicted_word", "entropy": "tf_entropy",
                           "entropy_top10": "tf_entropy_top10", "surprisal": "tf_surprisal",
                           "target_in_top10": "tf_target_in_top10", "perplexity_per_sentence": "tf_perplexity"}, inplace=True)
tf_output_merged_with_lemmas = pd.concat([en_merged, tf_output], axis=1)

df_en_meco = pd.read_csv('df_en_meco.csv', sep=';')

test = df_en_meco.merge(tf_output_merged_with_lemmas, how='left', on=['text_id', 'total_word_idx'])

# set of participants for which merge doesn't work
part_list = pd.unique(test[test['word_x'] != test['word_y']]['participant']) 
df_en_meco_1 = df_en_meco[df_en_meco['participant'].isin(part_list)]
df_en_meco_2 = df_en_meco[~df_en_meco['participant'].isin(part_list)]

tf_output_merged_with_lemmas_2 = tf_output_merged_with_lemmas.drop(['sent_id_and_idx', 'word_idx'], axis=1)
tf_output_merged_with_lemmas_1 = tf_output_merged_with_lemmas.drop(['sent_id_and_idx', 'word_idx'], axis=1)

# change the tf df for the first set of participants
tf_output_merged_with_lemmas_1_messed_up_part = tf_output_merged_with_lemmas_1[(tf_output_merged_with_lemmas_1['text_id'] == 3) 
                                                                   & (tf_output_merged_with_lemmas_1['total_word_idx'] > 148)].copy()
tf_output_merged_with_lemmas_1_not_messed_up_part = tf_output_merged_with_lemmas_1[((tf_output_merged_with_lemmas_1['text_id'] != 3) 
                                                                   | (tf_output_merged_with_lemmas_1['total_word_idx'] <= 148))].copy()
tf_output_merged_with_lemmas_1_messed_up_part['total_word_idx'] = tf_output_merged_with_lemmas_1_messed_up_part['total_word_idx'] - 1
tf_output_merged_with_lemmas_1 = pd.concat([tf_output_merged_with_lemmas_1_messed_up_part, tf_output_merged_with_lemmas_1_not_messed_up_part])

test1 = df_en_meco_1.merge(tf_output_merged_with_lemmas_1, how='left', on=['text_id', 'total_word_idx'])
test2 = df_en_meco_2.merge(tf_output_merged_with_lemmas_2, how='left', on=['text_id', 'total_word_idx'])

df_meco_tf = pd.concat([test1, test2])

# this needs to be 0
print(sum(df_meco_tf['word_x'] != df_meco_tf['word_y']))
df_meco_tf['word'] = df_meco_tf['word_x']
df_meco_tf.drop(['word_x', 'word_y'], axis=1, inplace=True)
df_meco_tf.to_csv('df_meco_tf.csv')


# MERGE POTSDAM DF WITH tf RESULTS

hi_merged = pd.read_csv("hi_merged.csv")
tf_output = pd.read_csv("HI_tf_test_results.csv")

# remove the end of sentence tag for the target word
tf_output = tf_output[tf_output['actual_word'] != '</s>'].reset_index()
tf_output.rename(columns={"correct": "tf_correct", "previous_word": "tf_previous_word",
                           "predicted_word": "tf_predicted_word", "entropy": "tf_entropy",
                           "entropy_top10": "tf_entropy_top10", "surprisal": "tf_surprisal",
                           "target_in_top10": "tf_target_in_top10", "perplexity_per_sentence": "tf_perplexity"}, inplace=True)
tf_output_merged_with_lemmas = pd.concat([hi_merged, tf_output], axis=1)

df_hi_potsdam = pd.read_csv('df_hi_potsdam.csv', ';')
df_potsdam_tf = df_hi_potsdam.merge(tf_output_merged_with_lemmas, how='left', on=['sent_id_and_idx', 'word_idx'])
df_potsdam_tf.drop(["entropy", "surprisal"], axis=1, inplace=True)

# this should be 0
print(sum(df_potsdam_tf['word_x'] != df_potsdam_tf['word_y']))
df_potsdam_tf['word'] = df_potsdam_tf['word_x']
df_potsdam_tf.drop(['word_x', 'word_y'], axis=1, inplace=True)
df_potsdam_tf.to_csv('df_potsdam_tf.csv')


# MERGE THE TWO DATAFRAMES INTO ONE SPECIFICALLY FOR R ANALYSIS
print(set(df_potsdam_tf.columns).difference(set(df_meco_tf.columns)))
print(set(df_meco_tf.columns).difference(set(df_potsdam_tf.columns)))

r_analysis_df =pd.concat([df_potsdam_tf, df_meco_tf], join='outer', axis=0, ignore_index=True)
r_analysis_df.to_csv('transformer_hi_en.csv')
