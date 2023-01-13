import math,random

## Define Stack class from <class list>
class Stack(list):
    push=list.append
    peek=lambda self: self[-1]
    isEmpty=lambda self:len(self)==0
    size=list.__len__    

## Define binominal tree from <class list>
class BinominalTree(list):
    def __init__(self):
        '''
        stocks: contains stock price of each node
        options: contains option value of each node
        '''
        super().__init__()
        self._stocks = []
        self._options = []
    
    def getStocks(self):
        return self._stocks
    
    def getOptions(self):
        return self._options

    def setParams(self,**kwargs):
        '''
        You mush specify the following parameters to start:
            S: stock price at time 0
            K: strike price of the option
            N: number of periods to the option expiration
            rtn: average annual return of the underlying asset
            sigma: annual volatility of the underlying asset
        '''
        self._price = kwargs['S']
        self._K = kwargs['K']
        self._N = kwargs['N']
        self._T = kwargs['T']
        self._rtn = kwargs['rtn']
        self._sigma = kwargs['sigma']

        ## interval: time interval of each period, in terms of a year
        self._interval = self._T / self._N
        self._stocks.append([self._price])

        ## u: the multiplier if stock price rises
        ## d: the multiplier if stock price falls
        ## p: the risk-neutral probability that the stock price will rise in the next period
        ## q: the risk-neutral probability that the stock price will fall in the next period
        ## r: discount rate
        self._u = math.exp(self._sigma * math.sqrt(self._interval))
        self._d = 1 / self._u
        self._p = (math.exp(self._rtn * self._interval) - self._d) / (self._u - self._d)
        self._q = 1 - self._p
        self._r = math.exp(-self._rtn * self._interval)
    
    def insertLayer(self):
        ## Calculate new stock prices based on the information of the previous period
        tempLayer = [[x*self._u,x*self._d] for x in self._stocks[-1]]
        ## Drop duplicates of stock prices (as p*q=1, there will be many duplicates each period)
        newLayer = [item[0] for item in tempLayer]+[tempLayer[-1][1]]
        self._stocks.append(newLayer)
    
    def getHeight(self):
        return len(self._stocks)
    
    ## Generate the binominal tree of the specified height
    def generateTree(self):
        height=self._N+1
        while self.getHeight()<height:
            self.insertLayer()
    
    ## Calculate the value of option on each node
    def valCal(self,isCall=True,isEU=True):
        '''
        isCall: whether the option is a call (True), or a put (False)
        isEU: whether the option is a European option (True), or an American option (False)
        '''
        
        ## We mush reversely deduct for values
        idx = self.getHeight() - 1
        while idx>=0:
            current = self._stocks[idx]
            ## Calculate the value if the option is executed immediately
            lastLayer = [max(x-self._K,0) if isCall else max(self._K-x,0) for x in current]
            if idx+1 == self.getHeight():
                ## At the bottom of the tree, the option values equals lastLayer
                self._options.append(lastLayer)
            
            else:
                ## option value of the previous layer
                previous = self._options[-1]

                ## Calculate values of option though backward deduction
                tempLayer = [
                    self._r * (self._q * previous[i] + self._p * previous[i-1])
                    for i in range(1,len(previous))
                ]

                ## Compare the value of immediate execution with the value calculated through backward deduction
                ## Appliable to American options
                optionLayer = [
                    lastLayer[i]
                    if lastLayer[i]>tempLayer[i] else tempLayer[i]
                    for i in range(len(current))
                ]
                if isEU:
                    self._options.append(tempLayer)
                else:
                    self._options.append(optionLayer)
            
            idx-=1
        
        self._options.sort(key=lambda x:len(x))
    
    ## Visulize the tree
    def printTree(self,printType="price"):
        '''
        printType: print stock prices or option values
        '''
        if printType=="price":
            iterObj=self._stocks
        elif printType=="value":
            iterObj=self._options
        else:
            pass
                
        idx=1
        ## Save the strings to print, which follows First-in-last-out principle in my codes
        strStack=Stack()

        for i in range(1, len(iterObj) + 1):
            spaceNum = 8
            spaces = " " * spaceNum  ## Set spaces for padding

            ## Format the numbers and generate strings for printing
            printStr = f"{len(iterObj) - idx:02d}: " \
                       + spaces * (idx - 1) \
                       + spaces.join(map(lambda x: f"{x:#.7g}", iterObj[-i]))
            size = len(iterObj[-i])
            strStack.push(printStr)
            idx += 1

            ## Generate strings of the branches
            jointStr = " " * (spaceNum * (idx - 1) + 3) + "/" \
                       + str(spaces + "\\" + " " * (spaceNum - 2) + "/") * (size - 2) \
                       + spaces + "\\"
            if idx <= len(iterObj):
                strStack.push(jointStr)
            else:
                strStack.push("\n")

        ## Print the tree
        while not strStack.isEmpty():
            s = strStack.pop()
            print(s)


def main():
    tree = BinominalTree()
    tree.setParams(S=50, K=45, N=7, T=1, sigma=0.08 * math.sqrt(12), rtn=0.12)
    tree.generateTree()
    tree.valCal(isCall=False)
    tree.printTree(printType="value")
    tree.printTree()


main()


## P.S. The original method I used is a tree-like class
## However, as the self-defined class is not as efficient as the python 'list' class, and include too many duplicates
## I turned to python 'list' class for help, and defined the 'BinominalTree' class as presented above.
## However, I still consider my previous work a not bad attempt.
class binaryTree(object):
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
        self.value = 0

    def setParams(self, **kwargs):
        self.price = kwargs['S']
        self.K = kwargs['K']
        self.N = kwargs['N']
        self.T = kwargs['T']
        self.rtn = kwargs['rtn']
        self.sigma = kwargs['sigma']
        self.interval = self.T / self.N

        self.u = math.exp(self.sigma * math.sqrt(self.interval))
        self.d = 1 / self.u
        self.p = (math.exp(self.rtn * self.interval) - self.d) / (self.u - self.d)
        self.q = 1 - self.p
        self.r = math.exp(-self.rtn * self.interval)

    ## Insert left child
    ## If leftChild not exists, create one
    ## Otherwise, create a new 'tree' and insert it between the original 2 nodes
    def insertLeft(self, newNode):
        if self.leftChild is None:
            self.leftChild = binaryTree(newNode)
            self.leftChild.price = self.price * self.u
        else:
            t = binaryTree(newNode)
            t.price = self.leftChild.price
            t.leftChild = self.leftChild
            t.leftChild.price = self.leftChild.price * self.u
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild is None:
            self.rightChild = binaryTree(newNode)
            self.rightChild.price = self.price * self.d
        else:
            t = binaryTree(newNode)
            t.price = self.rightChild.price
            t.rightChild = self.rightChild
            t.rightChild.price = self.rightChild.price * self.d
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def getRootPrice(self):
        return self.price

    def getRootVal(self):
        return self.value

    ## Get all stock prices, recursively calling getStocks() method
    ## Including too many duplicates!!!
    def getStocks(self):
        left = self.leftChild.getStocks() if self.leftChild else list()
        right = self.rightChild.getStocks() if self.rightChild else list()
        return left + [self.getRootPrice()] + right

    ## Get height of the tree, recursively calling getHeight() method
    def getHeight(self):
        left = self.leftChild.getHeight() if self.leftChild else 0
        right = self.rightChild.getHeight() if self.rightChild else 0
        return max(left, right) + 1

    ## Get all option values, recursively calling getOptionss() method
    ## Including too many duplicates!!!
    def getOptions(self):
        left = self.leftChild.getOptions() if self.leftChild else list()
        right = self.rightChild.getOptions() if self.rightChild else list()
        return left + [self.getRootVal()] + right

    ## Calculate the option values
    def getVal(self, isCall, isEU):
        '''
        isCall: whether the option is a call (True), or a put (False)
        isEU: whether the option is a European option (True), or an American option (False)
        '''

        ## We mush reversely deduct for values
        ## The logic of calculation is similar as the BinominalTree.valCal(),
        ## but this includes recursive calls
        if self.leftChild is None and self.rightChild is None:
            if isCall:
                self.value = max(self.price - self.K, 0)
            else:
                self.value = max(self.K - self.price, 0)
        else:
            val1 = self.r * (
                   self.p * self.leftChild.getVal(isCall, isEU)
                   + self.q * self.rightChild.getVal(isCall, isEU)
            )
            if isCall:
                val2 = max(self.price - self.K, 0)
            else:
                val2 = max(self.K - self.price, 0)

            if isEU:
                self.value = val1
            else:
                self.value = max(val1, val2)

        return self.value

    ## Generate tree
    def generateTree(self, maxH):
        n1 = random.random()
        n2 = random.random()
        h = maxH
        self.insertLeft(n1)
        self.insertRight(n2)

        if self.getHeight() <= maxH:
            self.leftChild.generateTree(h - 1)
            self.rightChild.generateTree(h - 1)