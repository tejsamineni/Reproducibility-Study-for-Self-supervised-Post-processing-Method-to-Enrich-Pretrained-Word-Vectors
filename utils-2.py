import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.auto import tqdm  # Auto-compatible tqdm
import torch

""" Data Reader """

def LoadDatasets(DataName):
    if DataName == "DBpedia":
        names = ["class", "title", "content"]
        train_csv = pd.read_csv("/content/drive/MyDrive/NLP assignment 3/ClfDatasets/DBpedia/train.csv", names=names)
        test_csv = pd.read_csv("/content/drive/MyDrive/NLP assignment 3/ClfDatasets/DBpedia/test.csv", names=names)
        shuffle_csv = train_csv.sample(frac=1)
        x_train = pd.Series(shuffle_csv["content"])
        y_train = pd.Series(shuffle_csv["class"]) - 1
        x_test = pd.Series(test_csv["content"])
        y_test = pd.Series(test_csv["class"]) - 1

        with open("/content/drive/MyDrive/NLP assignment 3/ClfDatasets/DBpedia/classes.txt", "r") as f:
            TopicList = {line.strip(): idx for idx, line in enumerate(f)}

        Idx2Topic = {v: k for k, v in TopicList.items()}
        return x_train, y_train, x_test, y_test, TopicList, Idx2Topic

    elif DataName == "AGNews":
        train_path = "/content/drive/MyDrive/NLP assignment 3/ClfDatasets/AGNews/AGNews/train.csv"
        test_path = "/content/drive/MyDrive/NLP assignment 3/ClfDatasets/AGNews/AGNews/test.csv"
        classes_path = "/content/drive/MyDrive/NLP assignment 3/ClfDatasets/AGNews/AGNews/classes.txt"

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found at {test_path}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found at {classes_path}")

        names = ["class", "title", "content"]
        train_csv = pd.read_csv(train_path, names=names)
        test_csv = pd.read_csv(test_path, names=names)

        shuffle_csv = train_csv.sample(frac=1)
        x_train = pd.Series(shuffle_csv["content"])
        y_train = pd.Series(shuffle_csv["class"]) - 1
        x_test = pd.Series(test_csv["content"])
        y_test = pd.Series(test_csv["class"]) - 1

        with open(classes_path, "r") as f:
            TopicList = {line.strip(): idx for idx, line in enumerate(f)}

        Idx2Topic = {v: k for k, v in TopicList.items()}
        return x_train, y_train, x_test, y_test, TopicList, Idx2Topic

    elif DataName == "IMDB":
        x_train, y_train, x_test, y_test = [], [], [], []
        # Load labeled training data
        with open("/content/drive/MyDrive/NLP assignment 3/ClfDatasets/IMDB/labeledTrainData.tsv", "r") as f:
            f.readline()
            for line in f:
                line = line.split("\t")
                x_train.append(line[2][1:-1])
                y_train.append(int(line[1]))

        # Load test data
        with open("/content/drive/MyDrive/NLP assignment 3/ClfDatasets/IMDB/testData.tsv", "r") as f:
            f.readline()
            for line in f:
                line = line.split("\t")
                x_test.append(line[1][1:-1])
                y_val = int(line[0].split("_")[-1][:-1])
                y_test.append(1 if y_val > 5 else 0)

        TopicList = {"negative": 0, "positive": 1}
        Idx2Topic = {0: "negative", 1: "positive"}
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), TopicList, Idx2Topic

    else:
        raise ValueError(f"Dataset {DataName} not supported.")

""" Data Preprocessing """

WordDict = {"<NONE>": 0, "<OOV>": 1}

def Preprocessing(string):
    """
    Preprocess the input string by removing unnecessary symbols and converting to lowercase.
    """
    string = re.sub(r"([\-,;\.!\?:\'\"/\|_#\$%\^\&\*~`\+=<>\(\)\[\]\{\}])", " \\1 ", string)
    return string.lower()

def DataProcessing(data, WordDict, WordCnt, TrainFlag):
    """
    Process text data into tokenized lists and update WordDict and WordCnt.
    """
    datas = []
    MaxSeqLen = 0
    pbar = tqdm(total=len(data), desc="Processing Data")
    for sent in data:
        sent = Preprocessing(sent)
        tokens = sent.split()

        if TrainFlag:
            for token in tokens:
                if token not in WordDict:
                    WordDict[token] = len(WordDict)
                    WordCnt[token] = 1
                else:
                    WordCnt[token] += 1

        MaxSeqLen = max(MaxSeqLen, len(tokens))
        datas.append(tokens)
        pbar.update(1)
    pbar.close()
    return datas, WordDict, WordCnt, MaxSeqLen

def Word2Tensor(Tokens, WordDict, MaxSeqLen):
    """
    Convert a list of tokens into a tensor of indices.
    """
    IdxTensor = torch.zeros(MaxSeqLen, dtype=torch.long)
    for i, token in enumerate(Tokens[:MaxSeqLen]):
        IdxTensor[i] = WordDict.get(token, WordDict["<OOV>"])
    return IdxTensor

def EmbeddingNumpy(Data, WordDict, MaxSeqLen):
    """
    Convert tokenized data into a NumPy array of embeddings.
    """
    Embed_np = np.zeros((len(Data), MaxSeqLen), dtype=np.int32)
    pbar = tqdm(total=len(Data), desc="Creating Embeddings")
    for i, tokens in enumerate(Data):
        Embed_np[i, :] = Word2Tensor(tokens, WordDict, MaxSeqLen)
        pbar.update(1)
    pbar.close()
    return Embed_np
