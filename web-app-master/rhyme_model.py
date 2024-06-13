import torch
from torch.nn import functional as F
import re
import string

torch.cuda.empty_cache()

from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM


def load_model(modelName, BATCH_SIZE, LANGUAGE, fullLines, genre):

    if modelName == 'rhyme_model':
        model_save_path = '/Users/elvinkaska/Downloads/flaskProject/models/' + genre + '_cross_roberta_batch' + str(BATCH_SIZE) + '_' + LANGUAGE
    else:
        model_save_path = '/Users/elvinkaska/Downloads/flaskProject/models/' + genre + '_cross_' + modelName + '_batch' + str(BATCH_SIZE) + '_' + LANGUAGE

    if not fullLines:
        model_path = model_save_path + '/'
    else:
        model_path = model_save_path + '_full_lines/'

    if modelName == 'bert':

        model = BertForMaskedLM.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    elif modelName == 'rhyme_model':

        model = RobertaForMaskedLM.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return tokenizer, model, device


def predict_rhyme(local_text, modelName='bert', topOne=True, topNum=10, BATCH_SIZE=16, LANGUAGE='eng', fullLines=False, genre='mixed'):

    tokenizer, model, _ = load_model(modelName=modelName, BATCH_SIZE=BATCH_SIZE, LANGUAGE=LANGUAGE, fullLines=fullLines, genre=genre)

    splitted = local_text
    # splitted = local_text.split('\n')
    splitted = [sentence.strip() for sentence in splitted]

    masked_sentences, not_masked_sentences = [], []

    for i, sentence in enumerate(splitted):

        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        loc_res = sentence.split(' ')

        if i == 2 or i == 3:

            loc_res[-1] = tokenizer.mask_token

        loc_str = ''
        for word in loc_res:

            if word not in string.punctuation:
                loc_str += ' ' + word

            else:
                loc_str += word

        line = loc_str.strip() + '\n'

        if i == 2 or i == 3:
            masked_sentences.append(line)
        else:
            not_masked_sentences.append(line)


    all_res = [not_masked_sentences[i] + masked_sentences[i] for i in range(2)]
    all_res_updated = [sentence.replace('\n', ' ').strip() for sentence in all_res]

    pred_words = []

    for i, sentence in enumerate(all_res_updated):

        sentence_toks = tokenizer.encode_plus(sentence, return_tensors="pt")

        outputs = model(**sentence_toks)

        softmax = F.softmax(outputs.logits, dim = - 1)

        mask_indexes = torch.where(sentence_toks["input_ids"][0] == tokenizer.mask_token_id)

        mask_word = softmax[0, mask_indexes, :]

        if topOne:
            top_word = torch.argmax(mask_word, dim=1)
            res = tokenizer.decode(top_word)
            res = res.replace(' ', '')
            new_sentence = sentence.replace(tokenizer.mask_token, res)
            pred_words.append(res)

        else:
            top_words = torch.topk(mask_word, topNum, dim=1)[1][0]
            loc_list = []
            for top_word in top_words:
                res = tokenizer.decode(top_word)
                res = res.replace(' ', '')
                new_sentence = sentence.replace(tokenizer.mask_token, res)
                loc_list.append(res)
            pred_words.append(res)

    result = format_result(splitted, pred_words)

    masked_sentences_output = []

    for item in all_res:

        loc_res = item.split('\n')
        del loc_res[-1]

        for item2 in loc_res:
            masked_sentences_output.append(item2)

    loc_sent = masked_sentences_output[1]
    masked_sentences_output[1] = masked_sentences_output[2]
    masked_sentences_output[2] = loc_sent

    return result, masked_sentences_output


def format_result(splitted, words):
    formated_data = []

    for i, sentence in enumerate(splitted):

        if i == 0 or i == 1:
            formated_data.append(sentence)

        else:
            loc_res = re.findall(r"[\w']+|[.,!?;]", sentence)

            if loc_res[-1] not in string.punctuation:
                loc_res[-1] = words[i - 2]

            else:
                loc_res[-2] = words[i - 2]

            loc_str = ''
            for word in loc_res:

                if word not in string.punctuation:
                    loc_str += ' ' + word

                else:
                    loc_str += word

            line = loc_str.strip()
            formated_data.append(line)

    return formated_data


def show_result(data):
    result = ''
    for sentence in data:
        result += sentence + '\n'
    return result


def get_res(text, language):
    res_rock_eng, masked_sentence_rock_eng = predict_rhyme(text, modelName='rhyme_model', topOne=True, BATCH_SIZE=32, LANGUAGE=language, fullLines=True)
    return show_result(res_rock_eng)
