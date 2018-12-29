"""
Class that reads the files
"""

import os.path  # enables operating system dependant functions, useful for traversing through files and directories

path = ""


cleanList = []  # The list that will be returned at the end.
length = 0  # The number of documents in the current file (length of docNumList)


def separate(new_path=""):
    __clearLists()  # clean the data structures.
    global path
    global textList
    global cleanList
    global length

    if new_path == "":  # There was a problem with the filepath given
        print("Error: No file path given.")
        return

    path = new_path

    for root, dirs, files in os.walk(path):  # traverses the given file path.
        text_file = open(os.path.join(path, files[0]), 'r', encoding="ISO-8859-1")  # The encoding of text files.
        textList = text_file.read().split("</TEXT>")  # Split at </TEXT> tag.
        text_file.close()
        del textList[-1]  # The last object in the list is the garbage past the final </TEXT> tag.

        length = len(textList)
        __takeText()
        __createCleanList()
    return cleanList


# Extract the clean text
def __takeText():
    global length
    global textList

    for i in range(length):
        text = textList[i].split("<TEXT>")[1].strip()
        # remove problematic or meaningless strings that clutter the corpus
        text = text.replace('\n', " ")
        text = text.replace('CELLRULE', " ")
        text = text.replace('TABLECELL', " ")
        text = text.replace('CVJ="C"', " ")
        text = text.replace('CHJ="C"', " ")
        text = text.replace('CHJ="R"', " ")
        text = ' '.join(text.split())
        textList[i] = text


# creates the dictionary to be returned
def __createCleanList():
    global length
    global docNumList
    global cleanList
    global textList
    global fileName

    for i in range(length):
        cleanList.append((docNumList[i], textList[i], fileName))


# Delete Reader's data structures
def __clearLists():
    global docNumList
    global textList
    global cleanList
    global length
    global path
    global headerList

    headerList = []
    docNumList = []
    textList = []
    cleanList = []
    length = 0
    path = ""


def reset():
    __clearLists()

