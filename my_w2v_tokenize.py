import re
import multiprocessing
import json
import gc
from os import popen
from sys import argv
from time import time

#import gensim
#from pympler import asizeof

KEYPHRASES_FILENAME = 'keyphrases_good.json'
TEXT_FILENAME = 'RC2017_1G.txt'
if len(argv) >= 2:
    TEXT_FILENAME = argv[1]

END_OF_LINES = {'.', ',', '!', '?', ':', ';'}

CHUNK_SIZE = 30000

REPLACE = {
    ' ': '_',
    '.': '_dot_',
    ',': '_comma_',
    ':': '_colon_',
    ';': '_semicolon_'
}
MIN_WINDOW_SIZE, MAX_WINDOW_SIZE = 1, 6


def load_json(filename):
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)

    return json_data


def load_normalized_keyphrases(filename):
    keyphrases = load_json(filename)
    normalized_keyphrases = {
        keyphrase: normalize_phrase(keyphrase) for keyphrase in keyphrases
    }
    del keyphrases
    gc.collect()

    return normalized_keyphrases


def count_lines(filename):
    command = 'wc -l {filename}'.format(filename=filename)
    command_output = popen(command).read()

    amount_of_lines = int(command_output.split()[0]) + 1
    return amount_of_lines


def normalize_line(line):
    line = line.rstrip('\r\n')
    line = line.lower()
    line = line.replace('?', ' . ')
    line = line.replace('!', ' . ')

    return line.split()


def normalize_phrase(phrase):
    for symbol, change in REPLACE.items():
        phrase = phrase.replace(symbol, change)

    return phrase




'''
def moving_window_preprocessing(text, min_size, max_size, normalized_keyphrases):
    def worker(amount, left, right):
        start = left
        while start < right:
            available = len(text) - start

            piece = text[start : start + max_size]

            for size in range(min(available, max_size), min_size - 1, -1):
                phrase = ' '.join(piece)

                if phrase in normalized_keyphrases:
                    amount[start] = size
                    start += size
                    break

                piece.pop()

            else:
                start += 1

    amount = multiprocessing.Array('B', len(text), lock=False)
    amount_of_processes = max(multiprocessing.cpu_count() - 1, 1)

    processes = []
    for index in range(amount_of_processes):
        processes.append(multiprocessing.Process(
            target=worker,

            args=(
                amount,

                len(text) // amount_of_processes * index,
                len(text) // amount_of_processes * (index + 1) if (index + 1 != amount_of_processes) else len(text)
            )
        ))

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    return amount
'''

def moving_window_preprocessing(text, min_size, max_size,
                                normalized_keyphrases):
    def worker(amount, left, right):
        start = left
        while start < right:
            available = len(text) - start

            piece = text[start: start + max_size]

            for size in range(min(available, max_size), min_size - 1, -1):
                phrase = ' '.join(piece)

                if phrase in normalized_keyphrases:
                    amount[start] = size
                    start += size
                    break

                piece.pop()

            else:
                start += 1

    # amount is shared array.
    # Othrwise the data used to exchange between processes and host is pickled,
    # which is slow
    '''
    amount = multiprocessing.Array('B', len(text), lock=False)
    amount_of_processes = max(multiprocessing.cpu_count() - 1, 1)

    processes = []
    for index in range(amount_of_processes):
        left = len(text) // amount_of_processes * index
        if index + 1 != amount_of_processes:
            right = len(text) // amount_of_processes * (index + 1)
        else:
            right = len(text)
        processes.append(multiprocessing.Process(target=worker,
                                                 args=(amount, left, right)))


    for process in processes:
        process.start()
    for process in processes:
        process.join()

    '''
    # removed multiprocessing:

    txt_len = len(text)
    amount = [0]*txt_len
    worker(amount, 0, txt_len)

    return amount

def moving_window_postprocessing(text, amount, min_size, normalized_keyphrases):
    output_text = []

    current = 0
    while current < len(text) - min_size + 1:
        if amount[current] != 0:
            phrase = ' '.join(text[current : current + amount[current]])
            output_text.append(normalized_keyphrases[phrase])

            current += amount[current]

        else:
            output_text.append(text[current])

            current += 1

    return output_text


def moving_window(text, min_size, max_size, normalized_keyphrases):
    amount = moving_window_preprocessing(text, min_size, max_size, normalized_keyphrases)
    output_text = moving_window_postprocessing(text, amount, min_size, normalized_keyphrases)

    del amount
    gc.collect()

    return output_text






def tokenize_keyphrases(text, normalized_keyphrases):
    last_line = text[-1]
    if last_line[-1] in END_OF_LINES:
        text[-1] = last_line[: -1]

    text = [word for word in text if word]

    output_text = moving_window(text, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, normalized_keyphrases)
    #output_text = ' '.join(output_text) + ' . '

    # with open('2.txt', 'w') as f:
    #    f.write(output_text)
    # exit(0)
    return output_text


normalized_keyphrases = load_normalized_keyphrases(KEYPHRASES_FILENAME)


def tokenizer(mystr):
    mystr = normalize_line(mystr)
    tokenized = tokenize_keyphrases(mystr, normalized_keyphrases)
    
    result = []
    
    for word in tokenized:
        if len(word)>0:
            if (word[-1] ==".") or (word[-1]==","):
                if len(word)>0:
                    result.append(word[:-1])
            else:
                result.append(word)
            
    return result


'''

def main():
    print('Keyphrases filename: {filename}'.format(filename=KEYPHRASES_FILENAME))
    print('Text filename: {filename}\n'.format(filename=TEXT_FILENAME))

    print('Loading keyphrases...')
    normalized_keyphrases = load_normalized_keyphrases(KEYPHRASES_FILENAME)
    print('Loading finished.\n')

    #model = gensim.models.Word2Vec(workers = (multiprocessing.cpu_count() - 1))
    first = True

    print('Counting lines...')
    total_lines = count_lines(TEXT_FILENAME)
    print('Total amount of lines: {amount}.\n'.format(amount=total_lines))

    iteration = 0
    end = False
    start_time = time()

    sample_file = open(TEXT_FILENAME, 'r')

    while not end:
        sample_words = []

        for index in range(CHUNK_SIZE):
            line = sample_file.readline()

            if not line:
                end = True
                break

            sample_words += normalize_line(line)

        if not sample_words:
            break

        #print('sample_words', asizeof.asizeof(sample_words))

        tokenized_sample_words = tokenize_keyphrases(sample_words, normalized_keyphrases)

        sentences = tokenized_sample_words.split(".")
        sentences = [sentence.split() for sentence in sentences if sentence]
        #sentences = gensim.models.word2vec.LineSentence(n_lines)

        token_count = sum(len(sentence) for sentence in sentences)

        print('Building dictionary...')
        if first:
            #model.build_vocab(sentences)
            first = False
        else:
            #model.build_vocab(sentences, update=True)
            first = False
        print('Building finished.')

        iteration += 1
        print('Lines processed: {current}/{total}, total: {percentage:.2f}%, time passed: {time:.2f}s.'.format(
            current=min(iteration * CHUNK_SIZE, total_lines),
            total=total_lines,
            percentage=min(iteration * CHUNK_SIZE, total_lines) * 100.0 / total_lines,
            time=time()-start_time
        ))

        print('Training...')
        #model.train(sentences, total_examples=token_count, epochs=model.iter)
        #model.save("model/reddit_w2v_model")
        if iteration * CHUNK_SIZE < total_lines:
            print('Moved to the next chunk.\n')
        else:
            print()

    sample_file.close()

    print("Training finished!")
    #model.save("model/reddit_w2v_model")

if __name__ == '__main__':
    main()
'''