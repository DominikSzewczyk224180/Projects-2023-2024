import tensorflow as tf

from transformers import glue_convert_examples_to_features

def create_tf_example(idx, features, label):
    """
    This method defines the structure of the Tensors used for Bert's classification model.
    :param idx: the id of each feature.
    :param features: the text of the feature.
    :param label: the label of the current feature.
    :return:
    tf.train.Example object with the data passed as an input parameters.
    """
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
        'sentence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features.encode('utf-8')])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))

    return tf_example


def convert_to_tfrecord(texts):
    """
    This method converts an array of sentences to tf.data.Dataset object.
    :param texts: array containing the texts.
    :return:
    tf.data.Dataset object containing all tf.train.Example objects.
    """
    records = []
    for idx, row in enumerate(texts):
        features, label = row, False
        records.append(create_tf_example(idx, features, label).SerializeToString())
    print(f"Fragment {texts} loaded for prediction")
    return tf.data.Dataset.from_tensor_slices(records)


def parse_example(example_proto):
    """
    This method parses the tfrecords.
    :param example_proto: tfrecord object containing the text
    :return:
    returns
    tf.Example object with the data.
    """
    feature_spec = {
        'idx': tf.io.FixedLenFeature([], tf.int64),
        'sentence': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    return tf.io.parse_single_example(example_proto, feature_spec)


def clean_string(features):
    """
    This method filters the text by removing the symbols: \n ... '
    :param features: str object representing the text
    :return:
    tf.strings object with preprocessed text in it
    """
    revised_sentence = tf.strings.regex_replace(features['sentence'], "\.\.\.", "", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\'", "'", replace_global=True)
    revised_sentence = tf.strings.regex_replace(revised_sentence, "\\n", "", replace_global=True)
    features['sentence'] = revised_sentence
    return features


def predict(model, tokenizer, txt, current_emotions):
    """
    This function encapsulates the logic required to predict an emotion using TFBertForSequenceClassification model.
    :param model: TFBertForSequenceClassification model
    :param tokenizer: BertTokenizer tokenizer
    :param txt: str representing the text
    :param current_emotions: the emotions that the model was trained on
    :return:
    str label of the predicted emotion
    """
    labels_encoded = []

    for iterator in range(len(current_emotions)):
        labels_encoded.append(str(iterator))

    kaggle_records = convert_to_tfrecord([txt])

    kaggle_parse_ds = kaggle_records.map(parse_example)

    kaggle_clean_ds = kaggle_parse_ds.map(lambda features: clean_string(features))

    kaggle_dataset = glue_convert_examples_to_features(examples=kaggle_clean_ds, tokenizer=tokenizer
                                                       , max_length=32, task='sst-2'
                                                       , label_list=labels_encoded)
    kaggle_dataset = kaggle_dataset.batch(1)

    y_pred = tf.nn.softmax(model.predict(kaggle_dataset, steps=1).logits)
    y_pred_argmax_new = tf.math.argmax(y_pred, axis=1)
    return current_emotions[y_pred_argmax_new]
