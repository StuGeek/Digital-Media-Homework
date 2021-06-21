#coding=utf-8
import nltk as nltk
import math as math
from tkinter import *

# 构建词袋模型计算相似度算法部分

# 对两个句子的单词列表进行分词获取词袋
def get_bags_of_word(word_lists):
    bags_of_word = set()
    # 将两个句子中出现过的单词不重复地添加到一个集合中，构成词袋
    for word in word_lists:
        bags_of_word.update(word)
    # 去掉词袋中的标点符号
    bags_of_word = bags_of_word - {',', '.', '，', '。', ':', '!'}
    return bags_of_word

# 处理词袋获得字典
def get_dictionary(bags_of_word):
    dictionary = dict()
    # 每个在句子对中出现过的单词对应一个数字
    for num, word in enumerate(bags_of_word):
        dictionary[word] = num
    return dictionary

# 根据词语出现频率TF值，处理单词列表和字典获取词袋模型向量
def get_TFvector(word_list, dictionary):
    TFvector = list()
    # 根据字典的关键字在单词列表中出现的频次计算次数
    for key in dictionary.keys():
        TFvector.append((dictionary[key], word_list.count(key)))
    return TFvector

# 打印词袋模型TF向量信息
def print_TFvector(TFvector):
    print(TFvector, end="")
    print("->[", end="")
    for i in range(len(TFvector)):
        if i != 0:
            print(", ", end="")
        print(TFvector[i][1], end="")
    print("]")

# 计算两个向量的余弦相似度
def get_cos_similarity(TFvector1, TFvector2):
    # 数量积
    scalar_product = 0
    TFvector1_length = 0
    TFvector2_length = 0
    for i in range(len(TFvector1)):
        scalar_product += TFvector1[i][1] * TFvector2[i][1]
        TFvector1_length += TFvector1[i][1] * TFvector1[i][1]
        TFvector2_length += TFvector2[i][1] * TFvector2[i][1]

    # 两个向量长度的乘积
    length_product = math.sqrt(TFvector1_length * TFvector2_length)
    return (scalar_product / length_product)

# 打印计算余弦相似度的过程
def print_cal_process(TFvector1, TFvector2):
    print("\nCalculate process:")
    scalar_product = 0
    TFvector1_length = 0
    TFvector2_length = 0

    print("scalar_product = ", end="")

    for i in range(len(TFvector1)):
        scalar_product += TFvector1[i][1] * TFvector2[i][1]
        if i != 0:
            print("+", end="")
        print("%d*%d"%(TFvector1[i][1], TFvector2[i][1]), end="")
    print(" = %d"%scalar_product)

    print("TFvector1_length = ", end="")

    for i in range(len(TFvector1)):
        TFvector1_length += TFvector1[i][1] * TFvector1[i][1]
        if i != 0:
            print("+", end="")
        print("%d^2"%(TFvector1[i][1]), end="")
    print(" = sqrt(%d)"%TFvector1_length)

    print("TFvector2_length = ", end="")

    for i in range(len(TFvector2)):
        TFvector2_length += TFvector2[i][1] * TFvector2[i][1]
        if i != 0:
            print("+", end="")
        print("%d^2"%(TFvector2[i][1]), end="")
    print(" = sqrt(%d)"%TFvector2_length)

    cos_similarity = scalar_product / math.sqrt(TFvector1_length * TFvector2_length)
    print("cos_similarity = scalar_product / (TFvector1_length * TFvector2_length) = %.6f"%cos_similarity)


# GUI图形界面部分

root = Tk()

# 设置句子1的标签和文本输入框
sent1_label = Label(root, font = 20, text = '句子1：')
sent1_label.place(relx = 0.05, rely = 0.05, relwidth = 0.1, relheight = 0.05)
sent1_input = Text(root, font = 10)
sent1_input.place(relx = 0.05, rely = 0.15, relwidth = 0.4, relheight = 0.6)

# 设置句子2的标签和文本输入框
sent2_label = Label(root, font = 20,text = '句子2：')
sent2_label.place(relx = 0.55, rely = 0.05, relwidth = 0.1, relheight = 0.05)
sent2_input = Text(root, font = 10) 
sent2_input.place(relx = 0.55, rely = 0.15, relwidth = 0.4, relheight = 0.6)

# 设置计算结果的标签和显示框
res_label = Label(root, text = '句子对的相似度（余弦相似度）为：', font = 20)
res_label.place(relx = 0.05, rely = 0.8, relwidth = 0.6, relheight = 0.05)
res_entry = Entry(root, font = 10)
res_entry.place(relx = 0.6,rely = 0.8, relwidth = 0.3,relheight = 0.05)

# 获取输入框中的内容计算结果
def cal_res():
    input1 = sent1_input.get(1.0, END)
    input2 = sent2_input.get(1.0, END)

    # 如果两个输入框都为空，那么相似度为1
    if (len(input1) == 1 and len(input2) == 1):
        return 1
    # 如果两个输入框只有一个空，那么相似度为0
    elif (len(input1) == 1 or len(input2) == 1):
        return 0

    # 将两个句子合并为句子对
    sentences = [input1, input2]
    # 将句子对进行分词，分成两个单词列表
    word_lists = [[word for word in nltk.word_tokenize(sentence)] for sentence in sentences]
    # 查看两个单词列表的信息
    print("\nword_lists:")
    print(word_lists)

    # 根据单词列表获取词袋
    bags_of_word = get_bags_of_word(word_lists)
    # 查看词袋信息
    print("\nbags of word:")
    print(bags_of_word)

    # 根据词袋获取字典
    dictionary = get_dictionary(bags_of_word)
    #查看字典信息
    print("\ndictionary:")
    print(dictionary)

    # 根据词频计数获取词袋模型向量
    TFvector1 = get_TFvector(word_lists[0], dictionary)
    TFvector2 = get_TFvector(word_lists[1], dictionary)
    # 查看词频向量信息
    print("\nTFvector1:")
    print_TFvector(TFvector1)
    print("TFvector2:")
    print_TFvector(TFvector2)

    # 计算句子对的余弦相似度
    cos_similarity = get_cos_similarity(TFvector1, TFvector2)
    # 打印计算过程
    print_cal_process(TFvector1, TFvector2)
    print("\n句子对的相似度（余弦相似度）为： %.6f。\n"%cos_similarity)
    return cos_similarity

# 按下计算按钮显示结果
def show_res():
    res_entry.delete(0, END)
    res_entry.insert(0, cal_res())

# 按下清除按钮将内容清空
def clear():
    sent1_input.delete(1.0, END)
    sent2_input.delete(1.0, END)
    res_entry.delete(0, END)

# 设置计算按钮
cal_button = Button(root, font = 5, text = '计算', command = show_res)
cal_button.place(relx = 0.46, rely = 0.3, relwidth = 0.075, relheight = 0.065)

# 设置清除按钮
clear_button = Button(root, font = 5, text = '清除', command = clear)
clear_button.place(relx = 0.46, rely = 0.4, relwidth = 0.075, relheight = 0.065)

# 设置退出按钮
quit_button = Button(root, font = 5, text = '退出', command = root.quit)
quit_button.place(relx = 0.46, rely = 0.5, relwidth = 0.075, relheight = 0.065)

root.title('句子对相似度计算')
root.geometry('800x600')
root.mainloop()
