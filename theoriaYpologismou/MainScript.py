import re
from typing import Any

class Dfa:
    num_of_states=0
    A=[]
    s=0
    f=[]
    states={}

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "device":
            print("device test")
        else:
            super(Dfa, self).__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def toString(self,item):
        a=''
        for i in item:
            a=a+str(i)
        return a

def readData(filename):
    f = open(filename, 'r')
    a=Dfa()

    data=f.readline()
    a.num_of_states=int(data.strip())
    # print('num_of_states',a.num_of_states)

    data=f.readline()
    temp=data.strip().split(' ')
    a.A=temp
    # print('alphavhto',a.A)

    data = f.readline()
    a.s = int(data.strip())
    # print('start',a.s)

    data = f.readline()
    temp = data.strip().split(' ')
    a.f = temp
    # print('finish',a.f)

    #creates a dictionary where each key represends a state and the equivalent value is a new dictionary
    #the key for the second dict represends the chars that it can read and the value is the outcome state
    #for example {'0':{'1':'2'}} means that from state 0 with input 1 we go to state 2
    for line in f:
        temp = line.strip().split(' ')
        if temp[0] in a.states:
            a.states[temp[0]][temp[1]]= temp[2]
            #if original state is in dictionary add the step {'0':{'1':'2'}} --> {'0':{'1':'2','0':'3'}}
        else:#create it all
            b = {temp[1]: temp[2]}
            a.states[temp[0]] = b
    #print('states:',a.states)
    f.close()
    return a

def checkCharInAlphabet(string,alphabet):
    for char in string:
        if char not in alphabet:
            return False
            break
    return True


def checkStringInAutomation(string, dfa):
    currentState=dfa.s
    flag=True
    for char in string:
        print('from state: ',currentState,'with ',char)
        if char in dfa.states[str(currentState)]:
            currentState=dfa.states[str(currentState)][char]
            print(' to state: ',currentState)
        else:
            flag=False
            break

    if currentState not in dfa.f:
        flag=False

    if flag is True:
        print('acceptable string\n')
    else:
        print('not acceptable string\n')


def main():
    #read the automation from file
    filename='dfa.txt'
    path='./InputFile/'+filename
    dfa=readData(path)
    #print(dfa.states)

    user=input("give a string to check or press \'exit\' to  exit\nit must only contains "+dfa.toString(dfa.A)+'\n\n')
    user = user.strip()
    while user!='exit':
        flag=checkCharInAlphabet(user,dfa.A)
        if(flag==False):
            print('input contains chars not in '+dfa.toString(dfa.A))
        else:
            checkStringInAutomation(user,dfa)
        user = input("give a string to check or press \'exit\' to  exit\nit must only contains " + dfa.toString(dfa.A) + '\n\n')
        user = user.strip()
    return 0

main()