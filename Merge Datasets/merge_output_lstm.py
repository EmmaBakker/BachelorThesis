import pandas as pd

# MERGE MECO DF WITH LSTM RESULTS

en_merged = pd.read_csv("en_merged.csv")
lstm_output = pd.read_csv("EN_lstm_test_results.csv")

# remove the end of sentence tag for the target word
lstm_output = lstm_output[lstm_output['actual_word'] != '</s>'].reset_index()
lstm_output.rename(columns={"correct": "lstm_correct", "previous_word": "lstm_previous_word",
                           "predicted_word": "lstm_predicted_word", "entropy": "lstm_entropy",
                           "entropy_top10": "lstm_entropy_top10", "surprisal": "lstm_surprisal",
                           "target_in_top10": "lstm_target_in_top10", "perplexity_per_sentence": "lstm_perplexity"}, inplace=True)
lstm_output_merged_with_lemmas = pd.concat([en_merged, lstm_output], axis=1)

df_en_meco = pd.read_csv('df_en_meco.csv', sep=';')

test = df_en_meco.merge(lstm_output_merged_with_lemmas, how='left', on=['text_id', 'total_word_idx'])

# set of participants for which merge doesn't work
part_list = pd.unique(test[test['word_x'] != test['word_y']]['participant']) 
df_en_meco_1 = df_en_meco[df_en_meco['participant'].isin(part_list)]
df_en_meco_2 = df_en_meco[~df_en_meco['participant'].isin(part_list)]

lstm_output_merged_with_lemmas_2 = lstm_output_merged_with_lemmas.drop(['sent_id_and_idx', 'word_idx'], axis=1)
lstm_output_merged_with_lemmas_1 = lstm_output_merged_with_lemmas.drop(['sent_id_and_idx', 'word_idx'], axis=1)

# change the lstm df for the first set of participants
lstm_output_merged_with_lemmas_1_messed_up_part = lstm_output_merged_with_lemmas_1[(lstm_output_merged_with_lemmas_1['text_id'] == 3) 
                                                                   & (lstm_output_merged_with_lemmas_1['total_word_idx'] > 148)].copy()
lstm_output_merged_with_lemmas_1_not_messed_up_part = lstm_output_merged_with_lemmas_1[((lstm_output_merged_with_lemmas_1['text_id'] != 3) 
                                                                   | (lstm_output_merged_with_lemmas_1['total_word_idx'] <= 148))].copy()
lstm_output_merged_with_lemmas_1_messed_up_part['total_word_idx'] = lstm_output_merged_with_lemmas_1_messed_up_part['total_word_idx'] - 1
lstm_output_merged_with_lemmas_1 = pd.concat([lstm_output_merged_with_lemmas_1_messed_up_part, lstm_output_merged_with_lemmas_1_not_messed_up_part])

test1 = df_en_meco_1.merge(lstm_output_merged_with_lemmas_1, how='left', on=['text_id', 'total_word_idx'])
test2 = df_en_meco_2.merge(lstm_output_merged_with_lemmas_2, how='left', on=['text_id', 'total_word_idx'])

df_meco_lstm = pd.concat([test1, test2])

# this needs to be 0
print(sum(df_meco_lstm['word_x'] != df_meco_lstm['word_y']))
df_meco_lstm['word'] = df_meco_lstm['word_x']
df_meco_lstm.drop(['word_x', 'word_y'], axis=1, inplace=True)
df_meco_lstm.to_csv('df_meco_lstm.csv')


# MERGE POTSDAM DF WITH LSTM RESULTS

hi_merged = pd.read_csv("hi_merged.csv")
lstm_output = pd.read_csv("HI_lstm_test_results.csv")

# remove the end of sentence tag for the target word
lstm_output = lstm_output[lstm_output['actual_word'] != '</s>'].reset_index()
lstm_output.rename(columns={"correct": "lstm_correct", "previous_word": "lstm_previous_word",
                           "predicted_word": "lstm_predicted_word", "entropy": "lstm_entropy",
                           "entropy_top10": "lstm_entropy_top10", "surprisal": "lstm_surprisal",
                           "target_in_top10": "lstm_target_in_top10", "perplexity_per_sentence": "lstm_perplexity"}, inplace=True)
lstm_output_merged_with_lemmas = pd.concat([hi_merged, lstm_output], axis=1)

df_hi_potsdam = pd.read_csv('df_hi_potsdam.csv', ';')
df_potsdam_lstm = df_hi_potsdam.merge(lstm_output_merged_with_lemmas, how='left', on=['sent_id_and_idx', 'word_idx'])
df_potsdam_lstm.drop(["entropy", "surprisal"], axis=1, inplace=True)

# this should be 0
print(sum(df_potsdam_lstm['word_x'] != df_potsdam_lstm['word_y']))
df_potsdam_lstm['word'] = df_potsdam_lstm['word_x']
df_potsdam_lstm.drop(['word_x', 'word_y'], axis=1, inplace=True)
df_potsdam_lstm.to_csv('df_potsdam_lstm.csv')


# MERGE THE TWO DATAFRAMES INTO ONE SPECIFICALLY FOR R ANALYSIS
print(set(df_potsdam_lstm.columns).difference(set(df_meco_lstm.columns)))
print(set(df_meco_lstm.columns).difference(set(df_potsdam_lstm.columns)))

r_analysis_df =pd.concat([df_potsdam_lstm, df_meco_lstm], join='outer', axis=0, ignore_index=True)
r_analysis_df.to_csv('lstm_hi_en.csv')
