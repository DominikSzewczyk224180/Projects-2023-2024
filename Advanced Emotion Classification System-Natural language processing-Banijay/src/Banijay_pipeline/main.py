import glob
import pandas as pd
import video_reader as v_r
from transformers import BertTokenizer, TFBertForSequenceClassification

model_path = './model/'
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

df_emotions = pd.read_csv('../../data/personal/Borislav/all_emotions.csv')
df = pd.read_csv(f'./new_file.csv')

data = {'video_filename': [],
        'fragment': [],
        'predictions': [],
        'text': []}

videos_location = 'C:/Users/Bobo/Downloads/task4/*.mov'
for i, video_filename in enumerate(glob.glob(videos_location)):
    predictions_per_fragment = v_r.predict_video(model, tokenizer, video_filename, (i + 13), df,
                                                 df_emotions['emotion'].unique()).values()

    for idx, (prediction, text) in enumerate(zip(*predictions_per_fragment)):
        data['video_filename'].append(video_filename)
        data['fragment'].append(idx)
        predic = prediction[0]
        if prediction[0] == 'love' or prediction[0] == 'joy':
            predic = 'happiness'
        data['predictions'].append(predic)
        data['text'].append(text)

    result_df = pd.DataFrame(data)
    result_df.to_csv('./result_right_fixed_4.csv', index=False)
