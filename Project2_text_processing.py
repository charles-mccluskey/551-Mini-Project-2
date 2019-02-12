#Starting with an empty file!
import os

def getAllData():
    posPath = "pos/"
    negPath = "neg/"
    posData = []
    negData = []
    for filename in os.listdir(posPath):
        posData.append(posPath+filename)

    for filename in os.listdir(negPath):
        negData.append(negPath+filename)

    dataDictionary = {"pos":posData,"neg":negData}
    return dataDictionary

def openComment(data,type,number):
    list = data.get(type)#retrieves either a positive or negative list of comments
    commentFile = list[number] #Gets the file we want to look at. Must be an int
    try:
        file = open(commentFile, "r") #Open the file in question
        comment = file.read() #read the file and store its string
        file.close() #close the file, since we can only have so many open at a time
    except:
        comment = None
    return comment #return the string that was contained within the file.

test = getAllData()
print(openComment(test,"neg",0))