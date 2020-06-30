#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon May  11 05:26:07 2020
@author: annapustova
"""

import os
import re
import csv
import time
import argparse
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.cost import cross_entropy_seq
from tensorlayer.models.seq2seq import Seq2seq
from pycontractions import Contractions

# Hyperparameters
epochs = 100
mini_batch = 64
deep_layers = 3
cell_units = 1024
embed_size = 1024
max_seq_length = 25
learn_rate = 0.001
learn_decay = 0.9
min_learn_rate = 0.0001
TOP_N = 3


# expand contractions
def exp_conts(text):
    text = text.lower()
    cont = Contractions(api_key="glove-twitter-100")
    text = cont.expand_texts(text)
    return text

# split into batches


def get_mini_batch(questions, responses, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index: start_index + batch_size]
        responses_in_batch = responses[start_index: start_index + batch_size]
        yield questions_in_batch, responses_in_batch

# strings into integers
# out for non-frequent words


def str_to_int(question, word_embed):
    question = exp_conts(question)
    return [word_embed.get(word, word_embed['<OUT>']) for word in question.split()]


def prepro_convos(dataset_name, path):
    if dataset_name == 'OES':
        questions = []
        responses = []
        conversations = []
        with open(path) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            prev_line = next(tsvreader)
            conv = [prev_line[3]]
            for line in tsvreader:
                if line[1] == prev_line[1]:
                    if line[2] == prev_line[2]:
                        prev_line[3] += " " + line[3]
                    else:
                        if conv[-1] != prev_line[3]:
                            conv.append(prev_line[3])
                        prev_line = line
                else:
                    conv.append(prev_line[3])
                    conversations.append(conv)
                    conv = [line[3]]
                    prev_line = line

        for conversation in conversations:
            for i in range(len(conversation) - 1):
                try:
                    questions.append(conversation[i])
                    responses.append(conversation[i+1])
                except:
                    pass
        return questions, responses
    elif dataset_name == 'KK':
        questions = []
        responses = []
        conversations = []
        with open(path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            conv = []
            for line in dialogue:
                conv.append(line[1])
            conversations.append(conv)

        for conversation in conversations:
            for i in range(len(conversation) - 1):
                try:
                    questions.append(conversation[i])
                    responses.append(conversation[i+1])
                except:
                    pass
        return questions, responses


def prepro_sets(dataset_name, path):
    # separate questions and responses
    questions, responses = prepro_convos(dataset_name, path)

    # preprocess the questions
    prepro_questions = []
    for question in questions:
        try:
            prepro_questions.append(exp_conts(question))
        except:
            print(question)

    # preprocess the responses
    prepro_responses = []
    for response in responses:
        prepro_responses.append(exp_conts(response))

    # filter out the questions and responses that exceed the threshold
    short_questions = []
    short_responses = []
    i = 0
    for question in prepro_questions:
        if 1 <= len(question.split()) <= max_seq_length:
            short_questions.append(question)
            short_responses.append(prepro_responses[i])
        i += 1
    prepro_questions = []
    prepro_responses = []
    i = 0
    for response in short_responses:
        if 1 <= len(response.split()) <= max_seq_length:
            prepro_responses.append(response)
            prepro_questions.append(short_questions[i])
        i += 1

    # word co-occurence dict
    word_occurence = {}
    for question in prepro_questions:
        for word in question.split():
            if word not in word_occurence:
                word_occurence[word] = 1
            else:
                word_occurence[word] += 1
    for response in prepro_responses:
        for word in response.split():
            if word not in word_occurence:
                word_occurence[word] = 1
            else:
                word_occurence[word] += 1

    # word embedding dictionary
    threshold_word_count = 15
    word_embed = {}
    word_number = 0
    for word, count in word_occurence.items():
        if count >= threshold_word_count:
            word_embed[word] = word_number
            word_number += 1

    # dialogue structure tokens
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    for token in tokens:
        word_embed[token] = len(word_embed) + 1

    # inverse the word_embed dictionary
    embed_to_word = {w_i: w for w, w_i in word_embed.items()}

    # add the fianle eos token to each response
    for i in range(len(prepro_responses)):
        prepro_responses[i] += ' <EOS>'

    # embed questions and responses
    # + replace filtered words with <OUT>
    embed_question_sequences = []
    for question in prepro_questions:
        ints = []
        for word in question.split():
            if word not in word_embed:
                ints.append(word_embed['<OUT>'])
            else:
                ints.append(word_embed[word])
        embed_question_sequences.append(ints)
    embed_response_sequences = []
    for response in prepro_responses:
        ints = []
        for word in response.split():
            if word not in word_embed:
                ints.append(word_embed['<OUT>'])
            else:
                ints.append(word_embed[word])
        embed_response_sequences.append(ints)

    # sort questions and responses by the length of questions
    sorted_prepro_questions = []
    sorted_prepro_responses = []
    for length in range(1, 25 + 1):
        for i in enumerate(embed_question_sequences):
            if len(i[1]) == length:
                sorted_prepro_questions.append(embed_question_sequences[i[0]])
                sorted_prepro_responses.append(embed_response_sequences[i[0]])
    vocab_size = len(word_embed) + 4

    return vocab_size, word_embed, embed_to_word, sorted_prepro_responses, sorted_prepro_questions

# split questions and responses into train and valid datsets


def split_dataset(test_size, sorted_prepro_questions, sorted_prepro_responses):
    tr_val_split = int(len(sorted_prepro_questions) * test_size)
    tr_questions = sorted_prepro_questions[tr_val_split:]
    tr_responses = sorted_prepro_responses[tr_val_split:]
    val_questions = sorted_prepro_questions[:tr_val_split]
    val_responses = sorted_prepro_responses[:tr_val_split]
    return tr_questions, tr_responses, val_questions, val_responses

# train


def train_model(model, optimizer, sorted_prepro_responses, sorted_prepro_questions, word_embed, vocab_size, dataset_name):
    tr_questions, tr_responses, val_questions, val_responses = split_dataset(
        test_size=0.15, sorted_prepro_responses=sorted_prepro_responses, sorted_prepro_questions=sorted_prepro_questions)
    model.train()
    batch_index_check_training_loss = 100
    batch_index_check_validation_loss = (
        (len(tr_questions)) // mini_batch // 2) - 1
    total_training_loss_error = 0
    list_validation_loss_error = []
    early_stopping_check = 0
    early_stopping_stop = 100
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_index, (questions_in_batch, responses_in_batch) in enumerate(get_mini_batch(tr_questions, tr_responses, mini_batch)):
            starting_time = time.time()
            pad_batch_questions = tl.prepro.pad_sequences(
                questions_in_batch, value=word_embed['<PAD>'])
            _target_seqs = tl.prepro.sequences_add_end_id(
                responses_in_batch, end_id=word_embed['<EOS>'])
            _target_seqs = tl.prepro.pad_sequences(
                _target_seqs, maxlen=max_seq_length)
            _decode_seqs = tl.prepro.sequences_add_start_id(
                responses_in_batch, start_id=word_embed['<SOS>'], remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(
                _decode_seqs, maxlen=max_seq_length)

            with tf.GradientTape() as tape:  # compute outputs
                output = model(
                    inputs=[pad_batch_questions, _decode_seqs])

                # compute perplexity and optimize
                output = tf.reshape(output, [-1, vocab_size])
                perplexity = cross_entropy_seq(
                    logits=output, target_seqs=_target_seqs)

                grad = tape.gradient(perplexity, model.all_weights)
                optimizer.apply_gradients(zip(grad, model.all_weights))

            total_training_loss_error += perplexity
            batch_time = time.time() - starting_time

            if batch_index % batch_index_check_training_loss == 0:
                print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Perplexity Score: {:>6.3f}, 100 Mini-batches Training Time: {:d} seconds'.format(epoch,
                                                                                                                                          epochs,
                                                                                                                                          batch_index,
                                                                                                                                          len(
                                                                                                                                              tr_questions) // mini_batch,
                                                                                                                                          total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                          int(batch_time * batch_index_check_training_loss)))
                total_training_loss_error = 0

            if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
                total_validation_loss_error = 0
                starting_time = time.time()
                for questions_in_batch, responses_in_batch in get_mini_batch(val_questions, val_responses, mini_batch):
                    pad_batch_questions = tl.prepro.pad_sequences(
                        questions_in_batch, value=word_embed['<PAD>'])
                    _target_seqs = tl.prepro.sequences_add_end_id(
                        responses_in_batch, end_id=word_embed['<EOS>'])
                    _target_seqs = tl.prepro.pad_sequences(
                        _target_seqs, maxlen=max_seq_length)
                    _decode_seqs = tl.prepro.sequences_add_start_id(
                        responses_in_batch, start_id=word_embed['<SOS>'], remove_last=False)
                    _decode_seqs = tl.prepro.pad_sequences(
                        _decode_seqs, maxlen=max_seq_length)
                    output = model(
                        inputs=[pad_batch_questions, _decode_seqs], seq_length=max_seq_length, start_token=word_embed['<SOS>'], top_n=1)
                    output = tf.reshape(output, [-1, vocab_size])

                    perplexity = cross_entropy_seq(
                        logits=output, target_seqs=_target_seqs)
                    total_validation_loss_error += perplexity

                ending_time = time.time()
                batch_time = ending_time - starting_time
                average_validation_loss_error = total_validation_loss_error / \
                    (len(val_questions) / mini_batch)
                print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                    average_validation_loss_error, int(batch_time)))
                optimizer.learning_rate = optimizer.learning_rate*learn_decay

                if optimizer.learning_rate < min_learn_rate:
                    optimizer.learning_rate = min_learn_rate
                list_validation_loss_error.append(
                    average_validation_loss_error)
                if average_validation_loss_error <= min(list_validation_loss_error):
                    print('Wow! My speaking skills have enchanced!')
                    early_stopping_check = 0
                    # save model as .npz
                    tl.files.save_npz(model.all_weights,
                                      name=dataset_name+'_chatbot_weights.npz')
                else:
                    print("Well, you need to optimize and I need to practice more.")
                    early_stopping_check += 1
                    if early_stopping_check == early_stopping_stop:
                        break
        if early_stopping_check == early_stopping_stop:
            print(
                "Early stopping")
            break
    print("Training has finished")


def begin_dialogue(model1, model2, word_embed, embed_to_word, checkpoint1, checkpoint2, top_n=TOP_N):
    # load model's weights and start the dialogue
    tl.files.load_and_assign_npz(name=checkpoint, network=model)
    tl.files.load_and_assign_npz(name=checkpoint2, network=model2)
    model = Seq2seqEnsemble(
        model1, model2, decoder_seq_length=max_seq_length, n_units=cell_units)
    model.eval()
    while(True):
        question = input('You: ')
        if question == 'Ciao':
            break
        question = str_to_int(question, word_embed)
        question = question + [word_embed['<PAD>']] * \
            (max_seq_length - len(question))
        predicted_response = model(
            inputs=[[question]], seq_length=max_seq_length, start_token=word_embed['<SOS>'], top_n=top_n).numpy()
        response = []
        for w_id in predicted_response[0]:
            word = embed_to_word[w_id]
            if word == 'i':
                response += [' I']
            elif word == '<EOS>':
                response += ['.']
            elif word == '<OUT>':
                response += ['out']
            else:
                response += [word]
            if word == '<EOS>':
                break
        print('KK + OES bi-RNN-LSTM: ' + ' '.join(response))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset1-name', default='KK', type=str,
                        choices=['OES', 'KK'], help='dataset1 name')
    parser.add_argument('--dataset1-path', type=str,
                        default='KK-bi-RNN-LSTM_data.json', help='KK dataset path')
    parser.add_argument('--checkpoint1', type=str, default='KK_weights.npz',
                        help='checkpoint path for the KK-bi-RNN-LSTM model')

    parser.add_argument('--dataset2-name', default='OES', type=str,
                        choices=['OES', 'KK'], help='dataset2 name')
    parser.add_argument('--dataset2-path', type=str,
                        default='oes-bi-RNN-LSTM_data.json', help='OES dataset path')
    parser.add_argument('--checkpoint2', type=str, default='OES_weights.npz',
                        help='checkpoint path for dataset2 model')

    parser.add_argument('--gpu', type=int,
                        help='GPU ID')

    args = parser.parse_args()

    dataset1_path = args.dataset1_path
    dataset2_path = args.dataset2_path
    dataset1_name = args.dataset1_name
    dataset2_name = args.dataset2_name
    checkpoint1 = args.checkpoint1
    checkpoint2 = args.checkpoint2

    vocab_size1, word_embed, embed_to_word, _, _ = prepro_sets(dataset1_name,
                                                               dataset1_path)
    vocab_size2, _, _, _, _ = prepro_sets(dataset2_name,
                                                 dataset2_path)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(len(gpus), "Physical GPU,")
    device = '/device:GPU:{}'.format(args.gpu) if gpus else '/device:CPU:0'
    with tf.device(device):
        model1 = Seq2seq(
            decoder_seq_length=max_seq_length,
            cell_enc=tf.keras.layers.LSTMCell,
            cell_dec=tf.keras.layers.LSTMCell,
            n_layer=deep_layers,
            n_units=cell_units,
            embedding_layer=tl.layers.Embedding(
                vocabulary_size=vocab_size1, embedding_size=embed_size),
        )
        model2 = Seq2seq(
            decoder_seq_length=max_seq_length,
            cell_enc=tf.keras.layers.LSTMCell,
            cell_dec=tf.keras.layers.LSTMCell,
            n_layer=deep_layers,
            n_units=cell_units,
            embedding_layer=tl.layers.Embedding(
                vocabulary_size=vocab_size2, embedding_size=embed_size),
        )

    start_chating(model1, model2, word_embed=word_embed, embed_to_word=embed_to_word,
                  checkpoint1=checkpoint1, checkpoint2=checkpoint2)


if __name__ == "__main__":
    main()
